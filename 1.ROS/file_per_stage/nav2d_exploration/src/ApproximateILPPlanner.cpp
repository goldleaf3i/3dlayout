// Approximate ILP Exploration Planner

#include "ApproximateILPPlanner.h"
#include <visualization_msgs/Marker.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <string.h>

#include <list>

#include <stdlib.h>     /* system, NULL, EXIT_FAILURE */
#include <boost/thread/mutex.hpp> // boost::mutex::scoped_lock

#include <angles/angles.h> // angles::normalize_angle

#define CONNECTION_REFUSED 111 // Error code when host is reachable
#define BS_ID 0 // BS ID for the python script
#define BIG_DISTANCE 10000000 // Big value for distance when not reachable

// Default strings
#define MAP_FRAME "map"

// Files to write for the python script
#define APPROX_FILE "approx.txt" // File for setting parameter (an integer)
#define COMM_GRAPH_FILE "G.txt" // matrix of communicability
#define DISTANCE_GRAPH_FILE "w_p.txt" // matrix of distances
#define FRONTIERS_FILE "F.txt" // list (same line) of frontier vertex ids (+1)
#define FRONTIERS_GAIN_FILE "FA.txt" // list (same line) of infogain corresponding to each vertex id in F (same order)
#define NOT_READY_DIST_FILE "notReadyDist.txt" // list of non-ready robots (ids) with list of distances between current pose and all vertices in the graph; final line: COMM <list of robot ids in communication with BS>
#define M_1_FILE "m_1.txt" // list of ready robot ids with current vertex id (one per each line)
#define CURR_WAITING_LIST_FILE "currWaitingList.txt" // list of frontier vertex ids with list of robot ids serving that frontier

// Files to read
#define M_OPT_FILE "m_opt.txt" // List of robot ids with associated the allocated vertex id (+1)
#define NEW_WAITING_LIST_FILE "waiting.txt" // list of frontier vertex ids with list of robot ids serving that frontier
#define COSTMIN_FILE "costMin.txt" // result of the solution (OK or Inf)
#define SOLUTION_NOT_FOUND "Inf" // String that indicates no more solution

// Files that need to be refreshed
#define CURRENT_TREE_FILE "currentTree.txt"

// Command to run
#define ILP_SCRIPT "approximate_deployement.py"
// Default path to the script
#define SCRIPT_PATH "/media/DATA/workspace/MRESim-master_4.0/MRESim-master_4.0/scripts/" 

#define SLASH "/"

/*
    @TODO lower the burden on the BS by asking distances to the robots
    @TODO Automatic connection of robots to the BS
    
*/

ApproximateILPPlanner::ApproximateILPPlanner()
{
  ROS_INFO_STREAM("CREATING Approximate PLANNER");
  
  // Create the class that keeps track of the robots poses
  mRobotList = new RobotList();
  
  // Some general common parameters
  ros::NodeHandle robotNode;
  robotNode.getParam("use_sim_time", mUseSimTime); // TODO Warning message in case it is not set
  robotNode.param("robot_id", mRobotID, BS_ID);
  robotNode.param("is_base_station", mIsBaseStation, false);
  robotNode.param("map_frame", mMapFrame, std::string(MAP_FRAME));

  ROS_INFO_STREAM("robot_id=" << mRobotID << ",is_base_station=" << mIsBaseStation << " sim_time=" << mUseSimTime << " map_frame=" << mMapFrame);
  
  // Parameters for the planner
  ros::NodeHandle navigatorNode("~/");
  navigatorNode.param("team_size", mTeamSize, 2);
  double percRobotsReadyThreshold;
  navigatorNode.param("perc_robots_ready_threshold", percRobotsReadyThreshold, 1.0);
  mNumRobotsReadyThreshold = ceil(percRobotsReadyThreshold * (mTeamSize-1));
  navigatorNode.param("communication_range", mCommunicationRange, 5.0);
  navigatorNode.param("communication_tolerance", mCommunicationTolerance, 0.5);
  navigatorNode.param("min_frontier_cluster_size", mMinClusterSize, 0.5);
  navigatorNode.param("neighborhood_tolerance", mNeighborhoodTolerance, 1.0);
  navigatorNode.param("safe_distance", mSafeDistance, 0.5);
  navigatorNode.param("distance_tolerance", mDistanceTolerance, 1.0);
  navigatorNode.param("angle_tolerance", mAngleTolerance, 0.5);
  navigatorNode.param("num_subset", mNumSubset, 2);
  // Setting paths for the script
  navigatorNode.param("root_path", mRootPath, std::string("/home/alberto/tmp/"));
  if (mRootPath.back() != '/')
      mRootPath.append(SLASH);
  navigatorNode.param("script_path", mScriptPath, std::string(SCRIPT_PATH));
  if (mScriptPath.back() != '/')
      mScriptPath.append(SLASH);
  
  ROS_INFO_STREAM("team_size=" << mTeamSize << ",NumRobotsReadyThreshold=" << mNumRobotsReadyThreshold  << ",min_target_area_size=" << mMinClusterSize << ",comm_range=" << mCommunicationRange << ",num_subset=" << mNumSubset);  

    // Visualization
  navigatorNode.param("visualize_markers", mVisualizeMarkers, true);
  if(mVisualizeMarkers)
  {
    mPoseRobotsPublisher = navigatorNode.advertise<visualization_msgs::Marker>("poses", 1, true);
    mTargetsPublisher = navigatorNode.advertise<visualization_msgs::Marker>("targets", 1, true);
    //cv::namedWindow("Map", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED);
  }
  
  if (mIsBaseStation)
  {
      /* Preparation of the files for the Python script */
      // Parameter for the approximation method
      std::ofstream myfile;
      ROS_INFO_STREAM("Writing file " << mRootPath + std::string(APPROX_FILE));
      myfile.open(mRootPath + std::string(APPROX_FILE));
      myfile << mNumSubset;
      myfile.close();
      
      // Remove files that should be read by this code to be safe
      remove(std::string(mRootPath + std::string(M_OPT_FILE)).c_str());
      remove(std::string(mRootPath + std::string(NEW_WAITING_LIST_FILE)).c_str());
      remove(std::string(mRootPath + std::string(COSTMIN_FILE)).c_str());
      remove(std::string(mRootPath + std::string(CURRENT_TREE_FILE)).c_str());
   
      /* Initializing the commanders to the robots */
      for (int i = 1; i < mTeamSize; i++)
      {
          int robotIndex = (!mUseSimTime) ? i : i - 1;
          mRobotsCommanders.push_back(new RobotNavigator(std::string("/robot_") + static_cast<std::ostringstream*>( &(std::ostringstream() << robotIndex) )->str() + std::string("/MoveTo/"), true)); // TODO parameter for the strings
          mRobotsCommanders.back()->waitForServer();
          ROS_INFO_STREAM("Creating commanders " << robotIndex);
      }         
    }
}

ApproximateILPPlanner::~ApproximateILPPlanner()
{
    mFrontiers.clear();
    mFrontiersClusters.clear();
    mCentroids.clear();

    mGraphVertices.clear();
    for (int i = 0; i < mCommunicationEdges.size(); i++)
        mCommunicationEdges[i].clear();
    mCommunicationEdges.clear();

    for (int i = 0; i < mDistanceEdges.size(); i++)
        mDistanceEdges[i].clear();

    mFrontier_infoGainList.clear();
    
    m_1.clear();
    notReady.clear();
    mCurrentWaitingList.clear();
        
  m_opt.clear();
  mRobotsCommanders.clear();
  
  delete mRobotList;

}

typedef std::multimap<double,unsigned int> Queue;
typedef std::pair<double,unsigned int> Entry;

int ApproximateILPPlanner::findExplorationTarget(GridMap* map, unsigned int start, unsigned int &goal)
{
    ROS_INFO_STREAM("findExplorationTarget");
    mMap = map;

    if (!mIsBaseStation)
    {
    }
    else // BS code
    {

      // Initialize offsets for 8-adjacency
      mOffset[0] = -1;          // left
      mOffset[1] =  1;          // right
      mOffset[2] = -map->getWidth();  // up
      mOffset[3] =  map->getWidth();  // down
      mOffset[4] = -map->getWidth() - 1;
      mOffset[5] = -map->getWidth() + 1;
      mOffset[6] =  map->getWidth() - 1;
      mOffset[7] =  map->getWidth() + 1;

      // Lock so that the feedback callback does not modify the data structures
      boost::mutex::scoped_lock lock(modify_mutex_);
      
      // Variable for filling just the relevant portion of the matrices
      int lastGraphSize = mGraphVertices.size();

      /* Initialization of the list of vertices of the graph with the current
        poses of the BS+robots */
      if (lastGraphSize == 0)
      {
        int vertexId;
        vertexId = insertVertexFromCell(start, false);
        if (vertexId != -1)
        {
          m_1.insert(std::make_pair(BS_ID, vertexId));
        }
        for (int i = 1; i < mTeamSize; i++)
        {
          vertexId = insertVertexFromRobotPose(i, false);
          if (vertexId != -1)
          {
            m_1.insert(std::make_pair(i, vertexId));
          }
          ROS_INFO_STREAM("[ApproximateILP] Inserting vertex " << vertexId << " for robot " << i);
        }
        previous_positions = m_1;
        ROS_INFO("[ApproximateILP] Using known positions of %d robots (BS+others).", (int)m_1.size());

        ROS_INFO_STREAM("Initialization of the robots pose");
      }

      
      if (mVisualizeMarkers)
        publishMarkers();
      
      /* Check status of the robots */
      std::set<unsigned int> robotsToPreempt;
      for (int i = 1; i < mTeamSize; i++)
      {
        if (mRobotsCommanders[i-1]->getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
        {
          // Robot arrived at destination
          robotsAtDestination.push_back(i);

          // Update not ready
          updateNotReady(i, mRobotsCommanders[i-1]->getResult()->final_pose);

          // Update ready
          updateReady();
        }
        else if (mRobotsCommanders[i-1]->getState() == actionlib::SimpleClientGoalState::ABORTED)
        {
          mBlacklistVertices.push_back(m_opt[i]);
          std::set<unsigned int> tmpRobotsToPreempt = robotsSharingFrontier(i);
          robotsToPreempt.insert(tmpRobotsToPreempt.begin(), tmpRobotsToPreempt.end());
          notReady.erase(i);
          
        }
      }
      
      for (auto robotId = robotsToPreempt.begin(); robotId != robotsToPreempt.end(); robotId++)
      {
          // TODO check if works removing just the robots with the branches

          if (mRobotsCommanders[*robotId-1]->getState() == actionlib::SimpleClientGoalState::ACTIVE)
          {
            ROS_INFO_STREAM("cancelling goal for robot " <<  *robotId);
            mRobotsCommanders[*robotId-1]->cancelAllGoals();
          }
          mRobotsCommanders[*robotId-1]->stopTrackingGoal();

          // Remove the topology of the tree (reinitialize the script)
          //remove(std::string(mRootPath + std::string(CURRENT_TREE_FILE)).c_str());
          
          // Clear relevant data structures
          notReady.erase(*robotId);
          robotsAtDestination.remove(*robotId);

          m_opt.erase(*robotId);
          // Reinitialization of the robots pose
          PoseList l = mRobotList->getRobots();
          int vertexId = insertVertexFromRobotPose(*robotId, true);
          if (vertexId != -1)
            m_1[*robotId] = vertexId; // TODO other value if robot pose not accessible?
          previous_positions.erase(*robotId);
          //ROS_INFO_STREAM("[ApproximateILP] Inserting vertex " << vertexId << " for robot " << *robotId << " at " << robot_pose);
          ROS_INFO("[ApproximateILP] Using known positions of %d robots (BS+others).", (int)m_1.size());
      }
      
      // Update the topology of the tree 
      if (robotsToPreempt.size() > 0)
        updateCurrentTree(robotsToPreempt);
      
      /* If at least mNumRobotsReadyThreshold robots are ready, compute a new allocation */
      if (m_1.size() > mNumRobotsReadyThreshold)
      {
        // Find frontiers to be added to the graph
        ROS_INFO_STREAM("Computing frontiers");

        findFrontiers(map, start);
        
        // Compute distances
        computeDistances(map, lastGraphSize);

        // Compute connectivity of the graph
        computeConnectivity(map, lastGraphSize);

        
        // Update notReady
        for (auto id = notReady.begin(); id != notReady.end(); id++)
        {
          if (id->second.size() != mGraphVertices.size())
          {
            PoseList l = mRobotList->getRobots();
            geometry_msgs::Pose2D robot_pose;
            robot_pose.x =  l[id->first].x;
            robot_pose.y =  l[id->first].y;
            robot_pose.theta = l[id->first].theta;              
            updateNotReady(id->first, robot_pose);
          }
        }
        
        // Preparation of the files
        writeFilesForILP();
        
        // Running the ILP model
        std::string command(std::string("python ") + std::string(mScriptPath) + std::string(ILP_SCRIPT) + std::string(" --root_path=") + mRootPath);
        int ret = system(command.c_str());
        printf ("The value returned was: %d.\n", ret);

        // Reading
        if (ret != 0 || !readFilesFromILP())
        {
            // TODO is it fine just to stop them? is it really true that when a new allocation is not found that is it over?
            ROS_INFO_STREAM("Return to the BS");
            
            // Return to the BS
            for (int i = 1; i < mTeamSize; i++)
            {
              ROS_INFO_STREAM("Robot id=" << i  << " to the goal with id=" << i << ",point=" << mGraphVertices[i]);

              // Get coordinates of the goal
              nav2d_navigator::MoveToPosition2DGoal goal;

              // Setting the goal
              goal.header.frame_id = mMapFrame;
              goal.target_pose.x = mGraphVertices[i].x;
              goal.target_pose.y = mGraphVertices[i].y;
              goal.target_pose.theta = mGraphVertices[i].theta;
              goal.target_distance = mDistanceTolerance;
              goal.target_angle = mAngleTolerance;

              if (mRobotsCommanders[i-1]->getState() == actionlib::SimpleClientGoalState::ACTIVE)
              {
                mRobotsCommanders[i-1]->cancelAllGoals(); 
                mRobotsCommanders[i-1]->stopTrackingGoal();
              }

              mRobotsCommanders[i-1]->sendGoal(goal);
              
            }
            return EXPL_FINISHED;
        }
          
        // 3. Assign robots to locations
        for (auto it = m_opt.begin(); it != m_opt.end(); it++)
        {
            ROS_INFO_STREAM("Assigning Robot id=" << it->first << " to the goal with id=" << it->second << ",point=" << mGraphVertices[it->second]);
            actionlib::SimpleClientGoalState clientState = mRobotsCommanders[it->first-1]->getState();
            if (clientState == actionlib::SimpleClientGoalState::ACTIVE)
              if (it->second == previous_m_opt[it->first])
              {
                ROS_INFO_STREAM("Same goal");
                continue;
              }
               
              else
              {
                mRobotsCommanders[it->first-1]->cancelAllGoals(); 
                mRobotsCommanders[it->first-1]->stopTrackingGoal();
              } 

            updateNotReady(it->first, mGraphVertices[m_1[it->first]]);

            // Setting the goal
            nav2d_navigator::MoveToPosition2DGoal goal;
            goal.header.frame_id = mMapFrame;  
            goal.target_pose.x = mGraphVertices[it->second].x;
            goal.target_pose.y = mGraphVertices[it->second].y;
            goal.target_pose.theta = mGraphVertices[it->second].theta;
            goal.target_distance = mDistanceTolerance;
            goal.target_angle = mAngleTolerance;

            // Cancel the goal previously assigned (if any)
            //if (mRobotsCommanders[it->first-1]->getState() == actionlib::SimpleClientGoalState::LOST)
            {
              ROS_INFO_STREAM("if assigning goal");              
              mRobotsCommanders[it->first-1]->sendGoal(goal, boost::bind(&ApproximateILPPlanner::doneCb, this, _1, _2),
                  boost::bind(&ApproximateILPPlanner::activeCb, this),
                  boost::bind(&ApproximateILPPlanner::feedbackCb, this, _1)); // TODO Offset
            }

          
            ROS_INFO_STREAM("Assigned robot=" << it->first << " to the goal with id=" << it->second << ",point=" << mGraphVertices[it->second]);         
            // Delete from list of robots ready
            previous_positions[it->first] = m_1[it->first];
            m_1.erase(it->first);
            
            // Delete from list of robots at destination but not ready
            robotsAtDestination.remove(it->first);

        }

        return EXPL_TARGET_SET;
      }
      else
        ROS_INFO_STREAM("Not enough robot ready (" << m_1.size() << ")");
    }
    return EXPL_WAITING;
}

int ApproximateILPPlanner::existingPoseInGraph(const geometry_msgs::Pose2D &pose)
{
    typedef VerticesList::value_type map_value_type;
    auto vertex_it = find_if(mGraphVertices.begin(), mGraphVertices.end(),[&pose,this](const map_value_type& vt)
        { return (fabs(vt.second.x - pose.x) < mNeighborhoodTolerance  && fabs(vt.second.y - pose.y) < mNeighborhoodTolerance); });
    if (vertex_it == mGraphVertices.end())
    {
      return -1;
    }
    else
    {
      return vertex_it->first;
    }
}

int ApproximateILPPlanner::insertVertexFromRobotPose(const unsigned int &robotId, const bool &existingCheck)
{
    PoseList l = mRobotList->getRobots();
    geometry_msgs::Pose2D robot_pose;
    while (!l.count(robotId))
    {
      ROS_INFO_STREAM("Cannot get the pose of robot " << robotId);
      l = mRobotList->getRobots();	
    }
    
    robot_pose.x = l[robotId].x;
    robot_pose.y = l[robotId].y;
    robot_pose.theta = l[robotId].theta;
    int vertexId;
    if (existingCheck)
    {
      vertexId = existingPoseInGraph(robot_pose);
      vertexId = (vertexId == -1) ? mGraphVertices.size() : vertexId;
    }
    else
    {
      vertexId = mGraphVertices.size();
    }
    mGraphVertices.insert(std::make_pair(vertexId, robot_pose));
    return vertexId;
}

int ApproximateILPPlanner::insertVertexFromCell(const unsigned int &cell, const bool &existingCheck)
{
    // add BS pose to the graph and to robots ready
    float x, y;
    if (mMap->getCoordinatesInWorld(x, y, cell))
    {
        geometry_msgs::Pose2D robot_pose;
        robot_pose.x = x;
        robot_pose.y = y;
        robot_pose.theta = 0.0;
        int vertexId;
        if (existingCheck)
        {
          vertexId = existingPoseInGraph(robot_pose);
          vertexId = (vertexId == -1) ? mGraphVertices.size() : vertexId;
        }
        else
        {
          vertexId = mGraphVertices.size();
        }

        mGraphVertices.insert(std::make_pair(vertexId, robot_pose));
        return vertexId;
    }
    else
      return -1;

}

std::set<unsigned int> ApproximateILPPlanner::robotsSharingFrontier(const unsigned int &robotId)
{
    std::set<unsigned int> robots;
    auto frontierIt = mCurrentWaitingList.begin();
    while (frontierIt != mCurrentWaitingList.end())
    {
        if (std::find(frontierIt->second.begin(), frontierIt->second.end(), robotId) != frontierIt->second.end())
        {
            robots.insert(frontierIt->second.begin(), frontierIt->second.end());
            mCurrentWaitingList.erase(frontierIt++);
        }
        else
          ++frontierIt;
    }
    return robots;
}

void ApproximateILPPlanner::findFrontiers(GridMap* map, unsigned int startCell)
{
  // TODO optimize by computing just the new frontiers ?

  // Clean list
  mFrontiers.clear();
  ROS_INFO_STREAM("findFrontiers");

  int k = 1, j, i;  // counters
  
  // Map dimension
  ROS_INFO_STREAM("size of the map "<< map->getHeight() << "x" << map->getWidth());

  //counter to stop the loop, because it's useless to check every cell in 
  // the circumference, if they are all not visible
  int ext_cells = k * 8;
  
  //tmpMap_ = map_.clone(); // DEBUG
  // BS position
  unsigned int rob_pos_x_, rob_pos_y_, cell_index; // TODO is it ok to have unsigned int? because of negative values when checking neighbors
  if(!map->getCoordinates(rob_pos_x_, rob_pos_y_, startCell))
  {
    //ROS_INFO_STREAM("not found, manually set");
    rob_pos_x_ = (unsigned int)(map->getWidth() / 2.0);
    rob_pos_y_ = (unsigned int)(map->getHeight() / 2.0);
  }

  //ROS_INFO_STREAM("rob_pox=" << rob_pos_x_ << "," << rob_pos_y_);    
  // find frontier cells, in a radial way, starting from robot position
  while (k < map->getHeight() || k < map->getWidth())
  {
    j = -k;
    while (j <= k)
    {
      i = -k;
      while (i <= k)
      {
        // if cell is frontier in freespace and is a frontier
        if (map->getIndex(rob_pos_x_+i, rob_pos_y_+j, cell_index) && map->isFree(cell_index) && map->isFrontier(cell_index))
        {
          mFrontiers.push_back(cell_index);
        }
        //else
        //  ext_cells--;
        // if on top or bottom, it has to check every cells
        if (j % k == 0 && j != 0)
          i++;
        // if in the middle, other inside cells already checked, so 
        // just most-left/right has to be checked
        else
          i = i + 2 * k;

      }
      j++;
    }
    k++;
    // stop criterion: if all cells in the circumference were not visible
    // or outside the map, stop COMMENTED BECAUSE OF SOME POSSIBLE ERRORS IN THE POSITIONING
    /*if (ext_cells == 0) // TOCHECK IF A CELL FALL OUTSIDE THE MAP
      break;
    else
      ext_cells = k * 8;*/
  }

  ROS_INFO_STREAM("#frontiers=" << mFrontiers.size());
  // find clusters
  if (!mFrontiers.empty())
  {
    FindClusters(map);
  }
}

void ApproximateILPPlanner::FindClusters(GridMap* map)
{
  // clear the list of clusters and centroids of clusters
  mFrontiersClusters.clear();
  mCentroids.clear();
  // Use of temporary list of frontiers
  Frontier tmpFrontiers(mFrontiers);
  std::list<double> centroidsX; // centroid for each cluster
  std::list<double> centroidsY;
  double centroidX, centroidY;
  unsigned int xCell, yCell;
  // Find clusters
  // Search for all adjacent cells starting from a cell, if not found, create a new cluster
  while (!tmpFrontiers.empty())
  {
    auto frontierIt = tmpFrontiers.begin(); 
    Frontier cluster;
    cluster.push_back(*frontierIt);
    centroidX = 0.0;
    centroidY = 0.0;
    tmpFrontiers.erase(frontierIt++);
    while (frontierIt != tmpFrontiers.end())
    {
        bool eliminated = false;
        for (auto clusterFrontierIt = cluster.begin(); clusterFrontierIt != cluster.end(); clusterFrontierIt++)
        {
            if (map->nearCells(*frontierIt, *clusterFrontierIt))
            {
                cluster.push_back(*frontierIt);

                if(map->getCoordinates(xCell, yCell, *frontierIt))
                {
                    centroidX += xCell;
                    centroidY += yCell;
                }
                tmpFrontiers.erase(frontierIt);

                frontierIt = tmpFrontiers.begin();
                eliminated = true;
                break;
            } 
        }
        if (!eliminated)
            frontierIt++;
    }
    mFrontiersClusters.push_back(cluster);        
    centroidsX.push_back(centroidX/cluster.size()); 
    centroidsY.push_back(centroidY/cluster.size());
  }
  ROS_INFO_STREAM("#clusters=" << mFrontiersClusters.size());
  
  // Find the frontier cell closest to the centroid for each cluster
  auto clustersIt = mFrontiersClusters.begin();
  auto centroidXIt = centroidsX.begin();
  auto centroidYIt = centroidsY.begin();

  unsigned int centroidIndex;
  while(clustersIt != mFrontiersClusters.end())
  {
    double tmpDistance, minDistance=BIG_DISTANCE; // TODO not use an arbitrary value
    for (auto clusterFrontierIt = clustersIt->begin(); clusterFrontierIt != clustersIt->end(); clusterFrontierIt++)
    {
        if(map->getCoordinates(xCell, yCell, *clusterFrontierIt))
        {
            double dx = *centroidXIt - xCell;
            double dy = *centroidYIt - yCell;
            tmpDistance = sqrt(dx * dx + dy * dy);
            if (tmpDistance < minDistance)
            {
                centroidIndex = *clusterFrontierIt;
                minDistance = tmpDistance;
            }
                
        }
    }
    mCentroids.push_back(centroidIndex);
    //ROS_INFO_STREAM("cluster size=" << clustersIt->size() << ",res=" << map->getResolution());
    if (clustersIt->size() > mMinClusterSize / map->getResolution())
    {
        typedef VerticesList::value_type map_value_type;
        
        // Find one of the neighbor unknown cells
         unsigned centroidX, centroidY, unknownX, unknownY, unknownId;
        for (int i = 0; i < 8; i++) // TODO parameter for setting the adjacency
        {
           unknownId = centroidIndex + mOffset[i];
           if (map->getData(unknownId) == -1)
           {

             if (map->getCoordinates(centroidX, centroidY, centroidIndex) && map->getCoordinates(unknownX, unknownY, unknownId))
             {
               break;
             }
           }
        }
        
        int dx = centroidX - unknownX, dy = centroidY - unknownY;
        unsigned int safeX = centroidX, safeY = centroidY, safeId;
        float x, y;
        // A certain distance from the frontier
        int safeDistance = mSafeDistance / map->getResolution();
        for (int l = 0; l < safeDistance; l++)
        {
          if(map->isFree(safeX, safeY) && map->getIndex(safeX, safeY, safeId) && map->getCoordinatesInWorld(x, y, safeId))
          {
            safeX += dx;
            safeY += dy;
          }
          else
            break;
        }
        geometry_msgs::Pose2D centroidPose; // TODO Use function to insert cell from pose
        centroidPose.x = x;
        centroidPose.y = y;
        centroidPose.theta = -angles::normalize_angle(atan2(dy, dx)); // TODO check angle
        unsigned int vertexId = existingPoseInGraph(centroidPose);
               
        if (vertexId == -1)
        {
            vertexId = mGraphVertices.size();
            mGraphVertices.insert(std::make_pair(vertexId, centroidPose));
            mFrontier_infoGainList.insert(std::make_pair(vertexId, clustersIt->size()));
        }
        else
        {
            mFrontier_infoGainList.insert(std::make_pair(vertexId, clustersIt->size()));
        }
        

    }
    clustersIt++;
    centroidXIt++;
    centroidYIt++;
  }
  ROS_INFO_STREAM("#centroids=" << mCentroids.size());
  
}


bool ApproximateILPPlanner::canCommunicate(const int &robotId)
{
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    struct sockaddr_in sin;
    sin.sin_family = AF_INET;
    sin.sin_port   = htons(65432);  // Could be anything
    std::string dest_address(std::string("192.168.1.") + static_cast<std::ostringstream*>( &(std::ostringstream() << robotId))->str());
    inet_pton(AF_INET, dest_address.c_str(), &sin.sin_addr); // TODO parameter for the prefix of the address

    if (connect(sockfd, (struct sockaddr *) &sin, sizeof(sin)) == -1)
    {
        ROS_WARN("Error connecting %s: %d (%s)\n", dest_address.c_str(), errno, strerror(errno));
    }

    return errno == CONNECTION_REFUSED;
}

void ApproximateILPPlanner::computeConnectivity(GridMap* map, const unsigned int &lastGraphSize)
{
    int graphSize = mGraphVertices.size();
    ROS_INFO_STREAM("Compute connectivity");
    mCommunicationEdges.resize(graphSize);
    unsigned int i, j; // Counters for the loops
    bool canCommunicate; // how to set the values of the communication matrix
    for (i = 0; i < mCommunicationEdges.size(); i++)
    {
        mCommunicationEdges[i].resize(graphSize);
    }
    
    // 2. Computation of the connectivity matrix
    //for (i = lastGraphSize; i < graphSize; i++)
    for (i = 0; i < graphSize; i++)
    {
        // If vertex blacklisted then set visibility to false
        if (std::find(mBlacklistVertices.begin(), mBlacklistVertices.end(), i) != mBlacklistVertices.end())
        {
          for (j = 0; j < graphSize; j++)
          {
            mCommunicationEdges[i][j] = false;
            mCommunicationEdges[j][i] = false;
          }
        }
        else
        {
          //for (j = (lastGraphSize > 0) ? 0 : i + 1; j < graphSize; j++)
          for (j = i + 1; j < graphSize; j++)
          {
              if (j != i)
              {
                  // Check whether the i vertex is reachable
                  if (setVerticesToUnreachable.find(i) != setVerticesToUnreachable.end() && !setVerticesToUnreachable[i])
                  {
                      canCommunicate = canProbablyCommunicate(map, mGraphVertices[i], mGraphVertices[j]);

                  }
                  else
                  {
                      canCommunicate = false;
                  }
                  mCommunicationEdges[i][j] = canCommunicate;
                  mCommunicationEdges[j][i] = canCommunicate;
              }
              
          }
        }
        mCommunicationEdges[i][i] = true;
    }
}

/* Implementation of Bresenham's line algorithm 
http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm */
bool ApproximateILPPlanner::canProbablyCommunicate(GridMap* map, geometry_msgs::Pose2D p1, geometry_msgs::Pose2D p2)
{   
      // Check how far they are
  double dx = p1.x - p2.x, dy = p1.y - p2.y;
  double distance = sqrt(dx * dx + dy * dy);
  if (distance > mCommunicationRange)
  {
    ROS_INFO_STREAM("Too far " << distance);
    return false;
  }
  unsigned int x1, y1, id1, x2, y2, id2;

  
  if(map->getIndexGivenWorld(p1.x, p1.y, id1) && map->getIndexGivenWorld(p2.x, p2.y, id2) && map->getCoordinates(x1, y1, id1) && map->getCoordinates(x2, y2, id2))
  {
      int delta_x((int)x2 - (int)x1);
      int delta_y((int)y2 - (int)y1);
      
      // if x1 == x2, then it does not matter what we set here
      signed char const ix((delta_x > 0) - (delta_x < 0));
      delta_x = std::abs(delta_x) << 1;


      // if y1 == y2, then it does not matter what we set here
      signed char const iy((delta_y > 0) - (delta_y < 0));
      delta_y = std::abs(delta_y) << 1;

      
      unsigned int cellsThreshold = std::ceil(mCommunicationTolerance / map->getResolution());
      unsigned int counter = 0;
      if (delta_x >= delta_y)
      {
        // error may go below zero
        int error(delta_y - (delta_x >> 1));

        while (x1 != x2)
        {
            if ((error >= 0) && (error || (ix > 0)))
            {
                error -= delta_x;
                y1 += iy;
            }
            // else do nothing

            if (map->getData(x1, y1) < 0 || map->getData(x1, y1) == 100)
            {
                counter++;
            }
            else
            {
                counter = 0;
            }

            if (counter < cellsThreshold)
            {
                error += delta_y;
                x1 += ix;
            }
            else
            {
                return false;
            }
        }
        return true;
      }
      else
      {
        // error may go below zero
        int error(delta_x - (delta_y >> 1));

        while (y1 != y2)
        {
            if ((error >= 0) && (error || (iy > 0)))
            {
                error -= delta_y;
                x1 += ix;
            }
            // else do nothing

            if (map->getData(x1, y1) < 0 || map->getData(x1, y1) == 100)
            {
                counter++;
            }
            else
            {
                counter = 0;
            }
            if (counter < cellsThreshold)
            {
                error += delta_x;
                y1 += iy;
            }
            else
            {
                return false;
            }
            
        }
      }
      return true;
  }
  else
  {
    return false;
  }

}

double ApproximateILPPlanner::computeDistance(GridMap* map, geometry_msgs::Pose2D start, geometry_msgs::Pose2D goal)
{
    unsigned int mapSize = map->getSize();
    double* plan = new double[mapSize];
    for(unsigned int i = 0; i < mapSize; i++)
    {
        plan[i] = -1;
    }

    // Initialize the queue with the robot position
    Queue queue;
    unsigned int startId, goalId;
    bool foundDistance = false;
    double distance;
    if (map->getIndexGivenWorld(start.x, start.y, startId) && map->getIndexGivenWorld(goal.x, goal.y, goalId))
    {
        Entry startPoint(0.0, startId);
        queue.insert(startPoint);
        plan[startId] = 0;

        Queue::iterator next;
        
        double linear = map->getResolution();
        
        int cellCount = 0;

        // Do full search with weightless Dijkstra-Algorithm
        while(!queue.empty())
        {
            cellCount++;
            // Get the nearest cell from the queue
            next = queue.begin();
            distance = next->first;
            unsigned int index = next->second;
            queue.erase(next);

            if(index == goalId) // We reached the goal point
            {
                foundDistance = true;
              
              break;
            }
            else // Add all adjacent cells
            {

  
              for(unsigned int it = 0; it < 8; it++)
              {
                unsigned int i = index + mOffset[it];
                if(map->isFree(i) && plan[i] == -1)
                {
                  queue.insert(Entry(distance+linear, i));
                  plan[i] = distance+linear;
                }
              }
            }
        }
        //ROS_INFO("Checked %d cells.", cellCount);  
        //ROS_INFO_STREAM("[ApproximateILPPlanner] Points " << start << "-" << goal << "has distance" << distance);
    }

    delete[] plan;

    if (!foundDistance)
        distance = BIG_DISTANCE;
    return distance;
}

void ApproximateILPPlanner::computeDistances(GridMap* map, const unsigned int &lastGraphSize)
{
    ROS_INFO_STREAM("Compute distances");
    mDistanceEdges.resize(mGraphVertices.size());
    for (int i = 0; i < mDistanceEdges.size(); i++)
        mDistanceEdges[i].resize(mGraphVertices.size());
    
    // 2. Computation of the distances matrix
    //for (unsigned int i = lastGraphSize; i < mGraphVertices.size(); i++)
    unsigned int graphSize = mGraphVertices.size();
    for (unsigned int i = 0; i < graphSize; i++)
    {
        mDistanceEdges[i][i] = 0.0;
        setVerticesToUnreachable[i] = false;
        //for (unsigned int j = (lastGraphSize > 0) ? 0 : i + 1; j < graphSize; j++)
        int sameDistanceCounter = 0;
        for (unsigned int j = i + 1; j < graphSize; j++)
        { 
            double distance = computeDistance(map, mGraphVertices[i], mGraphVertices[j]);  
           if (j < lastGraphSize && fabs(distance - mDistanceEdges[i][j]) < 0.01)
              sameDistanceCounter++;
            mDistanceEdges[i][j] = distance;
            mDistanceEdges[j][i] = distance;
        }
        
        if (sameDistanceCounter < lastGraphSize - i - 1)
        {
          auto blacklistedVertex = std::find(mBlacklistVertices.begin(), mBlacklistVertices.end(), i);
          if (blacklistedVertex != mBlacklistVertices.end())
            mBlacklistVertices.erase(blacklistedVertex);
        }
        int counter = 0;
        // TODO maybe it does not work because m_1 does not contain all the robots
        for (auto robotIt = m_1.begin(); robotIt != m_1.end(); robotIt++)
        {
            if (fabs(mDistanceEdges[i][robotIt->second] - BIG_DISTANCE) < 0.1)
                counter++;
            else
                break;
        }
        if (counter == m_1.size())
            setVerticesToUnreachable[i] = true;
    }

}

void ApproximateILPPlanner::writeFilesForILP()
{
    int i, j;
    // Write of the communication matrix
    std::ofstream outputFile, outputFile2;
    outputFile.open(mRootPath + std::string(COMM_GRAPH_FILE), std::ios::trunc);
    for (i = 0; i < mCommunicationEdges.size(); i++)
    {
        for (j = 0; j < mCommunicationEdges[i].size(); j++)
        {
            outputFile << mCommunicationEdges[i][j] << "\t";
        }
        outputFile << std::endl;
    }   
    outputFile.close();
 
    // Write distance matrix
    outputFile.open(mRootPath + std::string(DISTANCE_GRAPH_FILE), std::ios::trunc);
    for (i = 0; i < mDistanceEdges.size(); i++)
    {
        for (j = 0; j < mDistanceEdges[i].size(); j++)
        {
            outputFile << (int)mDistanceEdges[i][j] * 100 << "\t" << std::fixed; // TODO is this conversion really necessary?
        }
        outputFile << std::endl;
    }   
    outputFile.close();

    // Write ID of the frontiers and the corresponding information gain
    outputFile.open(mRootPath + std::string(FRONTIERS_FILE), std::ios::trunc);
    outputFile2.open(mRootPath + std::string(FRONTIERS_GAIN_FILE), std::ios::trunc);
    for(auto verticesIt = mFrontier_infoGainList.begin(); verticesIt != mFrontier_infoGainList.end(); verticesIt++)
    {
        outputFile << verticesIt->first + 1 << "\t"; // TODO Offset as a parameter
        outputFile2 << verticesIt->second << "\t"; // TODO separator as a parameter
    }
    outputFile.close();
    outputFile2.close();
 
    // Write robots ready   
    outputFile.open(mRootPath + std::string(M_1_FILE), std::ios::trunc);
    outputFile << BS_ID << "\t" << m_1[BS_ID] + 1 << std::endl;
    for (auto verticesIt = m_1.begin(); verticesIt != m_1.end(); verticesIt++)
    {
        if (verticesIt->first != BS_ID)
            outputFile << verticesIt->first << "\t" << verticesIt->second + 1 << std::endl; // TODO Offset as a parameter
    }
    outputFile.close();

    // Write robots not ready with distances
    outputFile.open(mRootPath + std::string(NOT_READY_DIST_FILE), std::ios::trunc);
    for (auto verticesIt = notReady.begin(); verticesIt != notReady.end(); verticesIt++)
    {
        outputFile << verticesIt->first;
        for (i = 0; i < (verticesIt->second).size(); i++)
        {
            outputFile << " " << (int)(verticesIt->second).at(i) * 100 << std::fixed;
        }
        outputFile << std::endl;
    }
    outputFile << "COMM";
    for (auto verticesIt = notReady.begin(); verticesIt != notReady.end(); verticesIt++)
    {
        if (mUseSimTime)
        {
            PoseList l = mRobotList->getRobots();
            if (canProbablyCommunicate(mMap, mGraphVertices[m_1[0]], l[verticesIt->first]))
                outputFile << " " << verticesIt->first;
        }
        else
        {
            if (canCommunicate(verticesIt->first))
                outputFile << " " << verticesIt->first;
        }
    }
    outputFile.close();
    
    // Write frontiers currently in the waiting list
    outputFile.open(mRootPath + std::string(CURR_WAITING_LIST_FILE), std::ios::trunc);
    for (auto verticesIt = mCurrentWaitingList.begin(); verticesIt != mCurrentWaitingList.end(); verticesIt++)
    {
        outputFile << verticesIt->first << std::endl;        
    }
    outputFile.close();   
}

bool ApproximateILPPlanner::readFilesFromILP()
{
    std::ifstream inputFile;
    // Check whether the solution is "OK" or "Inf"
    inputFile.open(mRootPath + std::string(COSTMIN_FILE));
    std::string result;
    if (inputFile.is_open())
    {
        inputFile >> result;
        ROS_INFO_STREAM("result of read " << result);
        if (result.compare(SOLUTION_NOT_FOUND) == 0)
            return false;
    }
    inputFile.close();
            
    // Update m_opt
    previous_m_opt = m_opt;
    inputFile.open(mRootPath + std::string(M_OPT_FILE));
    int i, goal;
    if (inputFile.is_open())
    {
        while (inputFile >> i >> goal)
        {
            ROS_INFO_STREAM("m_opt i=" << i << ",goal=" << goal);
            if (i != BS_ID)
                m_opt[i] = goal - 1;
        }
    }
    inputFile.close();
    
    // update mCurrentWaitingList
    inputFile.open(mRootPath + std::string(NEW_WAITING_LIST_FILE));
    if (inputFile.is_open())
    {
        mCurrentWaitingList.clear();
        std::string line;
        unsigned int frontierId;
        while (getline(inputFile, line))
        {
            std::istringstream is(line);
            unsigned int robotId;
            std::vector<unsigned int> robotIds;
            is >> frontierId;
            while (is >> robotId)
            {
                robotIds.push_back(robotId);
            }
            
            mCurrentWaitingList.insert(std::make_pair(frontierId,robotIds));
        }
    }
    inputFile.close();
    
    return true;
}

void ApproximateILPPlanner::doneCb(const actionlib::SimpleClientGoalState& state, const nav2d_navigator::MoveToPosition2DResultConstPtr& result)
{
    if (mRobotsCommanders[result->robot_id-1]->getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    {
      ROS_INFO_STREAM("DONE: Robot " << result->robot_id << " arrived at " << result->final_pose);
    }
    else if (mRobotsCommanders[result->robot_id-1]->getState() == actionlib::SimpleClientGoalState::ABORTED)
    {
      ROS_INFO_STREAM("Robot " << result->robot_id << " ABORTED");
    }
    else if (mRobotsCommanders[result->robot_id-1]->getState() == actionlib::SimpleClientGoalState::PREEMPTED)
      ROS_INFO_STREAM("Robot " << result->robot_id << " PREEMPTED");
      
}

void ApproximateILPPlanner::updateReady()
{
    ROS_INFO_STREAM("updateReady");
    // Check if a frontier can be removed from mCurrentWaitingList
    auto it = mCurrentWaitingList.begin();
    while (it != mCurrentWaitingList.end())
    {
        //ROS_INFO_STREAM("Frontier " << it->first);
        unsigned int numRobotsArrived = 0;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            if (std::find(robotsAtDestination.begin(), robotsAtDestination.end(), *it2) != robotsAtDestination.end())
            {
                numRobotsArrived++;
            }
            else
            {
                break;
            }
        }
        if (numRobotsArrived == it->second.size())
        {
            if (mFrontier_infoGainList.count(it->first))
              mFrontier_infoGainList.erase(it->first);
            mCurrentWaitingList.erase(it++);
        }
        else
        {
            ++it;
        }
    }
    
    // Check if a robot can be set to ready
    auto robotIt = robotsAtDestination.begin();
    while (robotIt != robotsAtDestination.end())
    {
        bool canSetRobotReady = true;
        for (auto it = mCurrentWaitingList.begin(); it != mCurrentWaitingList.end(); it++)
        {
            if (std::find(it->second.begin(), it->second.end(), *robotIt) != it->second.end())
            {
                canSetRobotReady = false;
                break;
            }
        }
        if (canSetRobotReady)
        {
            m_1.insert(std::make_pair(*robotIt,m_opt[*robotIt]));
            notReady.erase(*robotIt);
            robotsAtDestination.erase(robotIt++);
        }
        else
        {
            ++robotIt;
        }
            
    }
}


void ApproximateILPPlanner::updateNotReady(unsigned int robotId, geometry_msgs::Pose2D currentPosition)
{
    ROS_INFO_STREAM("updateNotReady from robot " << robotId);
    // Update distances
    if (notReady.count(robotId)) // If robotId exists
    {
        notReady[robotId].resize(mGraphVertices.size());
        for (auto it = mGraphVertices.begin(); it != mGraphVertices.end(); it++)
        {
            notReady[robotId].at(it->first) = computeDistance(mMap, currentPosition, it->second);
        }
    }
    else // create a new one
    {
        std::vector<double> distances(mGraphVertices.size());
        for (auto it = mGraphVertices.begin(); it != mGraphVertices.end(); it++)
        {
            distances.at(it->first) = computeDistance(mMap, currentPosition, it->second);
        }
        notReady.insert(std::make_pair(robotId, distances));
    }
}


void ApproximateILPPlanner::feedbackCb(const nav2d_navigator::MoveToPosition2DFeedbackConstPtr& feedback)
{
    ROS_INFO_STREAM("FEEDBACK: robot " << feedback->robot_id << " at " << feedback->current_pose);

    // Update distances of not ready
    if (mRobotsCommanders[feedback->robot_id-1]->getState() == actionlib::SimpleClientGoalState::ACTIVE)
    {
      boost::mutex::scoped_lock lock(modify_mutex_, boost::try_to_lock);
      if (lock.owns_lock())
        updateNotReady(feedback->robot_id, feedback->current_pose);   
    }
}

void ApproximateILPPlanner::activeCb()
{
}


cv::Mat ApproximateILPPlanner::convertOccupancyGridToImage(GridMap* map)
{
    map_ = cv::Mat(map->getHeight(), map->getWidth(),CV_8UC1);

    int channels = map_.channels();
    int nRows = map_.rows;
    int nCols = map_.cols;

    if (map_.isContinuous())
    {
        nCols *= nRows;
        nRows = 1;
    }

    int i, j;
    uchar* p;
    for( i = 0; i < nRows; ++i)
    {
        p = map_.ptr<uchar>(i);
        for ( j = 0; j < nCols; ++j)
        {
            if (mMap->getMap().data[i * nCols + j] == -1)
                p[j] = 0;
            else
                if (mMap->getMap().data[i * nCols + j] >=0 &&  mMap->getMap().data[i * nCols + j] < mMap->getLethalCost())
                    p[j] = 255;
                else if (mMap->getMap().data[i * nCols + j] >= mMap->getLethalCost())
                    p[j] = 127;
        }
    }


}

void ApproximateILPPlanner::drawMap()
{
    int key = 0;
    ROS_INFO_STREAM("map size " << mMap->getSize());

    map_ = convertOccupancyGridToImage(mMap);
    
    cv::imshow("Map", map_);
    key = cv::waitKey(1000/30);
}

void ApproximateILPPlanner::publishMarkers()
{
    visualization_msgs::Marker marker;
		marker.header.frame_id = mMapFrame;
		marker.header.stamp = ros::Time();
		marker.id = 0;
		marker.type = visualization_msgs::Marker::SPHERE_LIST;
		marker.action = visualization_msgs::Marker::ADD;
		marker.pose.position.x = 0;
		marker.pose.position.y = 0;
		marker.pose.position.z = 0;
		marker.pose.orientation.x = 0.0;
		marker.pose.orientation.y = 0.0;
		marker.pose.orientation.z = 0.0;
		marker.pose.orientation.w = 1.0;
		marker.scale.x = 0.1;
		marker.scale.y = 0.1;
		marker.scale.z = 0.1;
		marker.color.a = 1.0;
		marker.color.r = 0.0;
		marker.color.g = 1.0;
		marker.color.b = 0.0;
		marker.points.resize(mTeamSize);
		for (int i = 0; i < mTeamSize; i++)
		{
		  double x, y;
		  if (m_1.find(i) != m_1.end())
		  {
	      x = mGraphVertices[m_1[i]].x;
	      y = mGraphVertices[m_1[i]].y;
			}
			else
			{
	      x = mGraphVertices[previous_positions[i]].x;
	      y = mGraphVertices[previous_positions[i]].y;

			}
			marker.points[i].x = x;
			marker.points[i].y = y;
			marker.points[i].z = 0;
		}
		mPoseRobotsPublisher.publish(marker);
		
		if (m_opt.size() > 0)
		{
		  marker.header.frame_id = mMapFrame;
		  marker.header.stamp = ros::Time();
    	marker.type = visualization_msgs::Marker::CUBE_LIST;
    	marker.id = 0;
		  marker.color.a = 1.0;
		  marker.color.r = 1.0;
		  marker.color.g = 0.0;
		  marker.color.b = 0.0;
		  marker.points.resize(mTeamSize-1);
      for (int i = 1; i < mTeamSize; i++)
		  {
		    double x, y;
		    if (m_opt.find(i) != m_opt.end())
		    {
	        x = mGraphVertices[m_opt[i]].x;
	        y = mGraphVertices[m_opt[i]].y;
			  }
			  marker.points[i-1].x = x;
			  marker.points[i-1].y = y;
			  marker.points[i-1].z = 0;
		  }
		  mTargetsPublisher.publish(marker); 
		}
		// TODO edges
}

void ApproximateILPPlanner::updateCurrentTree(const std::set<unsigned int> &robotsToPreempt)
{
    // update current tree
    std::ifstream inputFile;
    inputFile.open(mRootPath + std::string(CURRENT_TREE_FILE));
    std::ostringstream os;
    int i, goal;
    if (inputFile.is_open())
    {
        while (inputFile >> i >> goal)
        {
            os << i << '\t';
            if (robotsToPreempt.count(i))
              os << m_1[i];
            else
              os << goal;
            os << std::endl;
        }
    }
    inputFile.close(); 
    std::ofstream outputFile;   
    outputFile.open(mRootPath + std::string(CURRENT_TREE_FILE), std::ios::trunc);
    outputFile << os.str();
    outputFile.close();
}
