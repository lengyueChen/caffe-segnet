#include <cfloat>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/intersection_over_union_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
template <typename Dtype>
class IntersectionOverUnionLayerTest : public CPUDeviceTest<Dtype>{
protected:
	IntersectionOverUnionLayerTest()
	:	blob_bottom_data_(new Blob<Dtype>(2,1,2,2)),
		blob_bottom_label_(new Blob<Dtype>(2,1,2,2)),
		blob_top_(new Blob<Dtype>(1,1,1,1)){
		//fill values	
		FillerParameter filler_param;
	    GaussianFiller<Dtype> filler(filler_param);
	    filler.Fill(this->blob_bottom_data_);
	    blob_bottom_vec_.push_back(blob_bottom_data_);
	    filler.Fill(this->blob_bottom_label_);
	    blob_bottom_vec_.push_back(blob_bottom_label_);
	    blob_top_vec_.push_back(blob_top_);
	}
	~IntersectionOverUnionLayerTest(){
		delete blob_bottom_data_;
		delete blob_bottom_label_;
		delete blob_top_;
	}
	Blob<Dtype>* const blob_bottom_data_;
  	Blob<Dtype>* const blob_bottom_label_;
  	Blob<Dtype>* const blob_top_;
  	vector<Blob<Dtype>*> blob_bottom_vec_;
  	vector<Blob<Dtype>*> blob_top_vec_;

};
	

TYPED_TEST_CASE(IntersectionOverUnionLayerTest, TestDtypes);


TYPED_TEST(IntersectionOverUnionLayerTest, TestSetup) {
  LayerParameter layer_param;
  IntersectionOverUnionLayer<TypeParam> layer(layer_param);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // check bottom[0]
  EXPECT_EQ(this->blob_bottom_data_->num(),2);
  EXPECT_EQ(this->blob_bottom_data_->channels(),1);  
  EXPECT_EQ(this->blob_bottom_data_->height(),2);
  EXPECT_EQ(this->blob_bottom_data_->width(),2);
  //check bottom[1]
  EXPECT_EQ(this->blob_bottom_label_->num(),2);
  EXPECT_EQ(this->blob_bottom_label_->channels(),1);
  EXPECT_EQ(this->blob_bottom_label_->height(),2);
  EXPECT_EQ(this->blob_bottom_label_->width(),2);
  //check top
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}


TYPED_TEST(IntersectionOverUnionLayerTest, TestForward){
	LayerParameter layer_param;
	IntersectionOverUnionLayer<TypeParam> layer(layer_param);
  /* Prediction
                image 0: 
                class1  [2      2]
                        [1      1]
       
                
                image 1:
                class1  [0      1]
                        [2      1]
                
             
-------------------------------------
    Ground truth :  
                image 0:
                                [2          1]
                                [0          0]
                image 1: 
                                [0          1]
                                [2          0]

		
	*/
	/*-----Prediction-------*/
	//image 0
  this->blob_bottom_data_->mutable_cpu_data()[0] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[2] = 1;
  this->blob_bottom_data_->mutable_cpu_data()[3] = 1;

  //image 1        
  this->blob_bottom_data_->mutable_cpu_data()[4] = 0;
  this->blob_bottom_data_->mutable_cpu_data()[5] = 1;
  this->blob_bottom_data_->mutable_cpu_data()[6] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[7] = 1;


  /*-----Ground truth-------*/
  //image 0
  this->blob_bottom_label_->mutable_cpu_data()[0]= 2;
  this->blob_bottom_label_->mutable_cpu_data()[1]= 1;
  this->blob_bottom_label_->mutable_cpu_data()[2]= 0;
  this->blob_bottom_label_->mutable_cpu_data()[3]= 0;
  //image 1
  this->blob_bottom_label_->mutable_cpu_data()[4]= 0;
  this->blob_bottom_label_->mutable_cpu_data()[5]= 1;
  this->blob_bottom_label_->mutable_cpu_data()[6]= 2;
  this->blob_bottom_label_->mutable_cpu_data()[7]= 0;
	

  	
	//Forward test
	layer.Forward(this->blob_bottom_vec_,this->blob_top_vec_);
	
	/* Expected output: 

	*/
	//EXPECT_NEAR(this->blob_top_->cpu_data()[0], (float)1/3, 1e-4);
}




TYPED_TEST(IntersectionOverUnionLayerTest, TestBackward){
        LayerParameter layer_param;
        IntersectionOverUnionLayer<TypeParam> layer(layer_param);
/* Prediction
                image 0: 
                class1  [2      2]
                        [1      1]
       
                
                image 1:
                class1  [0      1]
                        [2      1]
                
             
-------------------------------------
    Ground truth :  
                image 0:
                                [2          1]
                                [0          0]
                image 1: 
                                [0          1]
                                [2          0]

    
  */
  /*-----Prediction-------*/
  //image 0
  this->blob_bottom_data_->mutable_cpu_data()[0] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[1] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[2] = 1;
  this->blob_bottom_data_->mutable_cpu_data()[3] = 1;

  //image 1        
  this->blob_bottom_data_->mutable_cpu_data()[4] = 0;
  this->blob_bottom_data_->mutable_cpu_data()[5] = 1;
  this->blob_bottom_data_->mutable_cpu_data()[6] = 2;
  this->blob_bottom_data_->mutable_cpu_data()[7] = 1;


  /*-----Ground truth-------*/
  //image 0
  this->blob_bottom_label_->mutable_cpu_data()[0]= 2;
  this->blob_bottom_label_->mutable_cpu_data()[1]= 1;
  this->blob_bottom_label_->mutable_cpu_data()[2]= 0;
  this->blob_bottom_label_->mutable_cpu_data()[3]= 0;
  //image 1
  this->blob_bottom_label_->mutable_cpu_data()[4]= 0;
  this->blob_bottom_label_->mutable_cpu_data()[5]= 1;
  this->blob_bottom_label_->mutable_cpu_data()[6]= 2;
  this->blob_bottom_label_->mutable_cpu_data()[7]= 0;

        //Backward test
  vector<bool> propagate_down;
  layer.Backward(this->blob_top_vec_,propagate_down,this->blob_bottom_vec_);

         /* Expected output: 

        */
        //EXPECT_NEAR(this->blob_top_->cpu_data()[0], (float)1/3, 1e-4);
}





} //namespace caffe