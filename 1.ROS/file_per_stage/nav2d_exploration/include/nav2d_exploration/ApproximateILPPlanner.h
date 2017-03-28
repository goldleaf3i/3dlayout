#ifndef APPROXIMATEILPPLANNER_H_
#define APPROXIMATEILPPLANNER_H_

#include <nav2d_navigator/ExplorationPlanner.h>
#include <unordered_map>
#include <vector>

#include <actionlib/client/simple_action_client.h>
#include <nav2d_navigator/MoveToPosition2DAction.h>

#include <ios>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ApproximateILPPlanner : public ExplorationPlanner
{
public:
        ApproximateILPPlanner();
        ~ApproximateILPPlanner();
        
        int findExplorationTarget(GridMap* map, unsigned int start, unsigned int &goal);
        
private:
        // Types for the graph
        typedef std::unordered_map<unsigned int, geometry_msgs::Pose2D> VerticesList; // uniqueVertexID-Position
        typedef std::vector< std::vector<bool> > CommunicationMatrix;
        typedef std::vector< std::vector<double> > DistanceMatrix;

        // Types for the frontiers
        typedef std::list<unsigned int> Frontier;
        typedef std::vector<Frontier> FrontierList;
        typedef std::unordered_map<unsigned int, double> Frontier_InfoGainList; // uniqueFrontierVertexID-InfoGain
        
        // Type for the actionlib client
        typedef actionlib::SimpleActionClient<nav2d_navigator::MoveToPosition2DAction> RobotNavigator;
        
        // Methods
        void findFrontiers(GridMap* map, unsigned int startCell);
        void FindClusters(GridMap* map);
        void findCluster(GridMap* map, unsigned int startCell);
        bool canCommunicate(const int &robotId);
        void computeConnectivity(GridMap* map, const unsigned int &lastGraphSize);
        bool canProbablyCommunicate(GridMap* map, geometry_msgs::Pose2D p1Index, geometry_msgs::Pose2D p2Index);
        void computeDistances(GridMap* map, const unsigned int &lastGraphSize);
        double computeDistance(GridMap* map, geometry_msgs::Pose2D start, geometry_msgs::Pose2D goal);
        std::set<unsigned int> robotsSharingFrontier(const unsigned int &robotId);
    
        // Tools
        int existingPoseInGraph(const geometry_msgs::Pose2D &pose);
        int insertVertexFromRobotPose(const unsigned int &robotId, const bool &existingCheck);
        int insertVertexFromCell(const unsigned int &cell, const bool &existingCheck);
        
        // Update functions of data structures that are used by Python
        void updateReady();
        void updateNotReady(unsigned int robotId, geometry_msgs::Pose2D currentPosition);
        void updateCurrentTree(const std::set<unsigned int> &robotsToPreempt);
        
        // Wrappers for Python scripts
        void writeFilesForILP();
        bool readFilesFromILP();
        
        // Get feedback on current status of other robots
        void doneCb(const actionlib::SimpleClientGoalState& state, const nav2d_navigator::MoveToPosition2DResultConstPtr& result);
        void feedbackCb(const nav2d_navigator::MoveToPosition2DFeedbackConstPtr& feedback);
        void activeCb();

        // Visualization functions
        cv::Mat convertOccupancyGridToImage(GridMap* map);
        void drawMap();
        void publishMarkers();
        
        // Common parameters
        /// Is simulation? (to have a correct offset for robot IDs (Stage vs real robots))
        bool mUseSimTime;
        /// ID of the robot
        int mRobotID;
        /// Is the robot the BS?
        bool mIsBaseStation;
        /// Frame for the visualization markers
        std::string mMapFrame;

        // Parameters for the planner
        /// Team size (number of robots including BS in the counting)
        int mTeamSize;
        /// if #robots ready (including BS) > threshold, replan (should be < mTeamSize)
        int mNumRobotsReadyThreshold;
        // Communication parameters        
        double mCommunicationRange; // meter
        double mCommunicationTolerance; // meter (how many cells unknown or obstacle before saying that two points cannot communicate) TODO related to communication range
        // Selection of points as vertices of the graph parameters
        /// Minimum cluster of a size (meter)
        double mMinClusterSize;
        double mNeighborhoodTolerance;
        // Distance from the frontier
        double mSafeDistance;        
        // Motion parameters
        double mDistanceTolerance;
        double mAngleTolerance;
        // Async method parameters
        int mNumSubset; // parameter for choosing the number of subsets        
        std::string mRootPath; // path to the shared files
        std::string mScriptPath; // path to script path
        // Visualization
        bool mVisualizeMarkers;
        
        // Internal data structures
        // Markers Publishers
        ros::Publisher mPoseRobotsPublisher;
        ros::Publisher mTargetsPublisher;
        
        // Components
        /// List of poses of other robots
        RobotList *mRobotList;
        // Actionlib clients to send commands to robots (only for BS)
        std::vector<RobotNavigator* > mRobotsCommanders;
        GridMap* mMap; // last map of the environment

        /// Offset for the adjacency
        unsigned int mOffset[8];
        boost::mutex modify_mutex_; // for accessing common data structures

        // Frontier cells
        Frontier mFrontiers;
        FrontierList mFrontiersClusters;
        Frontier mCentroids;
        
        // Data for the scripts
        VerticesList mGraphVertices; // Vertices of the graph that contains initial positions of the robots/BS and frontiers (the IDs are incremental and start from 0)
        CommunicationMatrix mCommunicationEdges; // Line-of-sight communication matrix
        std::unordered_map<unsigned int, bool> setVerticesToUnreachable; // true if no global path plan can be found from the current vertices of the robots, false otherwise
        DistanceMatrix mDistanceEdges; // Distance matrix (meters)
        Frontier_InfoGainList mFrontier_infoGainList; // list of current frontiers (verted ID) with info gain (size of the cluster in number of cells)
        std::unordered_map<unsigned int, unsigned int> previous_positions; // list of robots with their previous position (vertex id)        
        std::unordered_map<unsigned int, unsigned int> m_1; // list of robots ready with their current position (vertex id)
        std::unordered_map<unsigned int, std::vector<double> > notReady; // list of pairs of robotID-list of distances that are not ready (that is, they could be at the destination, but the branch towards a frontier is still not set)
        std::unordered_map<unsigned int, std::vector<unsigned int> > mCurrentWaitingList; // List of frontiers IDs that are currently "served" by robots with the related robots that are currently going to the respective positions
        std::list<unsigned int> robotsAtDestination; // temporary List of robotIDs that arrived at destination (maybe not ready yet) 
        std::unordered_map<unsigned int, unsigned int> m_opt; // For each robot a vertex is assigned (robotID-vertexID), including BS (both IDs start from 0)
        std::unordered_map<unsigned int, unsigned int> previous_m_opt;
        std::list<unsigned int> mBlacklistVertices;
        
        cv::Mat map_; // Occupancy grid canvas FOR DEBUGGING TODO remove
};

#endif // APPROXIMATEILPPLANNER_H_
