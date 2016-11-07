#include <utility>
#include <vector>
#include <functional>
#include <cfloat>
#include <iostream>
#include <cstdio>

//#include "caffe/intersection_over_union_layer.hpp"
//#include "caffe/util/math_functions.hpp"
//#include "caffe/vision_layers.hpp"
//#include "caffe/loss_layers.hpp"
//#include "caffe/layer.hpp"


#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe{

template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
	
}

template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->count()/bottom[0]->shape(1), bottom[1]->count())
		<< "Number of labels must match number of predictions; "
      	<< "e.g., if prediction shape is (N, C, H, W), "
      	<< "label count (number of labels) must be N*H*W."
      	<< "bottom[0] N*C*H*W: " << bottom[0]->count()
      	<< "bottom[0] C: " << bottom[0]->count()
      	<< "bottom[1] N*1*H*W: " << bottom[1]->count();
}

template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){

	Dtype num_class = this->layer_param_.intersection_over_union_param().num_class();

	const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	const Dtype* bottom_label= bottom[1]->mutable_cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int num = bottom[0]->num();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();

	float IUscore = 0.0;
	// C_i  number of correctly classified pixels in class i. 
	int C_i = 0;
	// G_i  total number of pixels whose label is i
	int G_i = 0;
	// P_i: total number of pixels whose prediction is i
	int P_i = 0;


	for (int class_idx = 0; class_idx < num_class; class_idx++){
		C_i=0;
		G_i=0;
		P_i=0;
		//calculate C_i
		for(int n = 0; n < num; n++){
			for(int i = 0; i < height*width;i++){ 
				const int idx = i+n*height*width;
				if( bottom_data[idx] == bottom_label[idx] &&bottom_data[idx]==class_idx ) {
					C_i++;
				}
				if(bottom_label[idx] == class_idx ){
					G_i++;
				}
				if(bottom_data[idx]== class_idx){
					P_i++;
				}
			}       
		}

		std::cout<<"fwd class :"<< class_idx <<std::endl;
		std::cout<<"C_i: "<<C_i<<std::endl;
		std::cout<<"G_i: "<<G_i<<std::endl;
		std::cout<<"P_i: "<<P_i<<std::endl;

		IUscore +=  (float)C_i/(G_i + P_i - C_i);
		std::cout<<"IUscore: "<<IUscore<<std::endl;
	}
	
	std::cout<<"FWD DEBUG1" <<std::endl;
	*top_data = 1 - IUscore / num_class;
	std::cout<<"FWD DEBUG2" <<std::endl;
	std::cout<<*top_data<<std::endl;
}



template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom){

		Dtype num_class = this->layer_param_.intersection_over_union_param().num_class();



        const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        
        const Dtype* bottom_label= bottom[1]->mutable_cpu_data();
        
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        

        const int num = bottom[0]->num();
        
        const int height = bottom[0]->height();
        const int width = bottom[0]->width();

        /*
        std::cout<< "Num: "<< bottom[0]->num() << std::endl;
        std::cout<< "Classes:"<< bottom[0]->channels() << std::endl;
        std::cout<< "Height:"<< bottom[0]->height() << std::endl;
        std::cout<< "Width:"<< bottom[0]->width() << std::endl;
        */

        
        // C_i  number of correctly classified pixels in class i. 
        double C_i = 0;
        
        // G_i  total number of pixels whose label is i
        double G_i = 0;
        
        // P_i: total number of pixels whose prediction is i
        double P_i = 0;
        

        for (int class_idx = 0; class_idx < num_class; class_idx++){
        		//label i, predict i
                C_i=0;
                vector<int> ii;
				//label i, predict j
                G_i=0;
                vector<int> ij;
                //label j, predict i
                P_i=0;
                vector<int> ji;
                
                for(int n = 0; n < num; n++){
					for(int i = 0; i < height*width;i++){        
						const int idx = i+n*height*width;
						if( bottom_data[idx] == bottom_label[idx] &&bottom_data[idx]==class_idx ) {
							C_i++;
							G_i++;
							P_i++;
							ii.push_back(idx);
							continue;
						}

						if( bottom_label[idx] == class_idx ) {
							G_i++;	
							ij.push_back(idx);
						}
						if( bottom_data[idx] == class_idx ) {
							P_i++;	
							ji.push_back(idx);
						}
					}
                }
                
                std::cout<<"bwd class :"<< class_idx <<std::endl;
				std::cout<<"C_i "<< C_i<<std::endl;
				std::cout<<"G_i "<< G_i<<std::endl;
				std::cout<<"P_i "<< P_i<<std::endl;


				double gradient_C_i_constant = (double)((G_i + P_i - 2*C_i));
                double gradient_C_i =  (-1)*gradient_C_i_constant/((double)(pow(gradient_C_i_constant+C_i,2)));
                double gradient_G_i_P_i = C_i/((double)(pow(G_i + P_i - C_i,2)));

                std::cout<<"gradient_C_i_constant "<< gradient_C_i_constant<<std::endl;
                std::cout<<"gradient_C_i "<< gradient_C_i<<std::endl;
                std::cout<<"gradient_gradient_G_i_P_iC_i_constant "<<gradient_G_i_P_i <<std::endl;

                std::cout<<"gradient_C_i  : "<<gradient_C_i <<std::endl;
                for(int i = 0;i < ii.size();i++){
                        int idx = ii[i];
                        std::cout<<idx<<std::endl;
                        bottom_diff[idx] += gradient_C_i;
                }
                std::cout<<"gradient_G_i_P_i (ij) : "<<gradient_G_i_P_i <<std::endl;
                for(int i = 0;i < ij.size();i++){
                        int idx = ij[i];
                        std::cout<<idx<<std::endl;
                        bottom_diff[idx] += gradient_G_i_P_i;
                }

                std::cout<<"gradient_G_i_P_i (ji): "<<gradient_G_i_P_i <<std::endl;
                for(int i = 0;i < ji.size();i++){
                        int idx = ji[i];
                        std::cout<<idx<<std::endl;
                        bottom_diff[idx] += gradient_G_i_P_i;
                }
        }

        //test
        std::cout<<bottom_diff[0]<<std::endl;
        std::cout<<bottom_diff[1]<<std::endl;
        std::cout<<bottom_diff[2]<<std::endl;
        std::cout<<bottom_diff[3]<<std::endl;
        std::cout<<bottom_diff[4]<<std::endl;
		std::cout<<bottom_diff[5]<<std::endl;
		std::cout<<bottom_diff[6]<<std::endl;
		std::cout<<bottom_diff[7]<<std::endl;

}



INSTANTIATE_CLASS(IntersectionOverUnionLayer);
REGISTER_LAYER_CLASS(IntersectionOverUnion);
} //namespace caffe
                                                        