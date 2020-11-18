#include <iostream>
#include <dirent.h>
#include <boost/program_options.hpp>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
//#include <pcl/common/point_operators.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/search/organized.h>
#include <pcl/search/octree.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/surface/gp3.h>
#include <pcl/io/vtk_io.h>
#include <pcl/filters/voxel_grid.h>

#include <fstream>
#include <cstring>
#include <cstdlib>
#include <pcl/ModelCoefficients.h>
#include <pcl/visualization/pcl_visualizer.h>



using namespace pcl;
using namespace std;

namespace po = boost::program_options;

pcl::PointCloud<pcl::PointXYZI>::Ptr filterCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, 
                                                pcl::PointIndices::Ptr              indices, 
                                                bool                                negative)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::ExtractIndices<pcl::PointXYZI> extract;
    extract.setInputCloud(cloud);
    extract.setIndices(indices);
    extract.setNegative(negative);
    extract.filter(*cloud_filtered);

    return cloud_filtered;
}



void doRansacPlane(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, 
                   pcl::PointIndices::Ptr              inliers, 
                   pcl::ModelCoefficients::Ptr         coefficients,
                   double                              distance_threshold)
{
    int max_iterations = 200;
    pcl::SACSegmentation<pcl::PointXYZI> seg;

    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(distance_threshold);
    seg.setMaxIterations(max_iterations);
    seg.setInputCloud(cloud);
    seg.segment(*inliers, *coefficients);
}

Eigen::Matrix4f CreateRotateMatrix(Eigen::Vector3f before,Eigen::Vector3f after)
{
    before.normalize();
    after.normalize();
  
    float angle = acos(before.dot(after));
    Eigen::Vector3f p_rotate =before.cross(after);
    p_rotate.normalize();
  
    Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
    rotationMatrix(0, 0) = cos(angle) + p_rotate[0] * p_rotate[0] * (1 - cos(angle));
    rotationMatrix(0, 1) = p_rotate[0] * p_rotate[1] * (1 - cos(angle) - p_rotate[2] * sin(angle));//这里跟公式比多了一个括号，但是看实验结果它是对的。
    rotationMatrix(0, 2) = p_rotate[1] * sin(angle) + p_rotate[0] * p_rotate[2] * (1 - cos(angle));
  
  
    rotationMatrix(1, 0) = p_rotate[2] * sin(angle) + p_rotate[0] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 1) = cos(angle) + p_rotate[1] * p_rotate[1] * (1 - cos(angle));
    rotationMatrix(1, 2) = -p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
  
  
    rotationMatrix(2, 0) = -p_rotate[1] * sin(angle) +p_rotate[0] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 1) = p_rotate[0] * sin(angle) + p_rotate[1] * p_rotate[2] * (1 - cos(angle));
    rotationMatrix(2, 2) = cos(angle) + p_rotate[2] * p_rotate[2] * (1 - cos(angle));
  
    return rotationMatrix;
}

vector<float> matrix2angle(Eigen::Matrix4f rotateMatrix)
{
	float sy = (float)sqrt(rotateMatrix(0,0) * rotateMatrix(0,0) + rotateMatrix(1,0)*rotateMatrix(1,0));
	bool singular = sy < 1e-6; // If
	float x, y, z;
	if (!singular)
	{
		x = (float)atan2(rotateMatrix(2,1), rotateMatrix(2,2));
		y = (float)atan2(-rotateMatrix(2,0), sy);
		z = (float)atan2(rotateMatrix(1, 0), rotateMatrix(0, 0));
	}
	else
	{
		x = (float)atan2(-rotateMatrix(1, 2), rotateMatrix(1, 1));
		y = (float)atan2(-rotateMatrix(2, 0), sy);
		z = 0;
	}
	vector<float> i;
	i.push_back((float)(x * (180.0f / M_PI)));
	i.push_back((float)(y * (180.0f / M_PI)));
	i.push_back((float)(z * (180.0f / M_PI)));
	return i;
}
 
int main(int argc, char** argv)
{
	///The file to read from.
	string indir;

	///The file to output to.
	string outdir;

	// Declare the supported options.
	po::options_description desc("Program options");
	desc.add_options()
		//Options
		("indir", po::value<string>(&indir)->required(), "the dir to read a point cloud from")
		("outdir", po::value<string>(&outdir)->required(), "the dir to write the DoN point cloud & normals to")
		;
	// Parse the command line
	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, desc), vm);

	// Print help
	if (vm.count("help"))
	{
		cout << desc << "\n";
		return -1;
	}

	// Process options.
	po::notify(vm);

	//读取pcd文件列表，按照字典顺序
	struct dirent **namelist;
	int n;
	std::vector<std::string> files;
	n = scandir(indir.c_str(),&namelist,0,alphasort);
	if(n >0)
	{ 
	  int index=0;

	  while(index < n)
	  {
	      if (strncmp(strrchr(namelist[index]->d_name,'.'),".pcd",4)==0) {
	        files.push_back(namelist[index]->d_name);
	      }
	      free(namelist[index]);
	      index++;
	  }
	  free(namelist);
	}
	std::cout<<"LOG::total read pcdfile "<<files.size() << endl;

	for(int i = 0; i < files.size(); i++) {
		string infile=indir;
		infile.append("/");
		infile.append(files[i]);

		string outfile=outdir;
		outfile.append("/");
		outfile.append(files[i]);
		
		string outtext=outdir;
		outtext.append("/");
		outtext.append(files[i]);
		outtext.append(".txt");

		ofstream textfile;
		textfile.open(outtext);
		std::cout<<"LOG::process pcdfile "<<infile<<endl;

		pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);

		if (pcl::io::loadPCDFile<pcl::PointXYZI>(infile.c_str(), *cloud) == -1) //* 读入PCD格式的文件，如果文件不存在，返回-1
		{
			PCL_ERROR("Couldn't read file \n"); //文件不存在时，返回错误，终止程序。
			continue;
		}
		textfile << "==================================\nLoaded "
			<< cloud->points.size()
			<< " data points from "<<infile
			<< std::endl;

		pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
		pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_withoutground(new pcl::PointCloud<pcl::PointXYZI>);
		pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ground(new pcl::PointCloud<pcl::PointXYZI>);
		

		//发现实际地面
		doRansacPlane(cloud,inliers,coefficients,0.025);

		//生成实际地面法线
		Eigen::Vector3f  v1;    
		v1 << coefficients->values[0], coefficients->values[1], coefficients->values[2];
		textfile<<"current normals:"<<endl<<v1<<endl;
		//生成标准地面法线
		Eigen::Vector3f  v2; 
		v2 << 0,0,1;
		textfile<<"standard normals:"<<endl<<v2<<endl;

		//计算旋转矩阵
		Eigen::Matrix4f rotation=CreateRotateMatrix(v1,v2);

		textfile<<"rotation matrix:"<<endl<<rotation<<endl;

		//得到旋转角
		vector<float> angle=matrix2angle(rotation);

		textfile<<"rotation angle:"<<angle[0]<<"\t"<<angle[1]<<"\t"<<angle[2]<<endl;

		pcl::PointCloud<pcl::PointXYZI>::Ptr newCloud(new pcl::PointCloud<pcl::PointXYZI>);

		//实施旋转
		pcl::transformPointCloud(*cloud, *newCloud, rotation);

		//去掉nan
		std::vector<int> mapping;
		pcl::removeNaNFromPointCloud(*newCloud, *newCloud, mapping);
		textfile<<"after remove nan remain "<<newCloud->points.size()<<" points"<<endl;

		//去掉地面
		for (int i=1;i<50;i++)
		{
			doRansacPlane(newCloud,inliers,coefficients,0.012);
			cloud_withoutground = filterCloud(newCloud, inliers, true);
			*newCloud=*cloud_withoutground;
		}

	    pcl::PCDWriter writer;
		writer.write(outfile.c_str(),*cloud_withoutground,true);
	      
		textfile << "Write  "
			<< cloud_withoutground->points.size()
			<< " data points into "<<outfile
			<< std::endl;

		textfile.close();
	}


	


	return 0;
}