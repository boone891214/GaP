export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
docker build  --build-arg http_proxy=http://11.162.93.61:3128 --build-arg https_proxy=http://11.162.93.61:3128 . --network=host -t transformer_pyt 
