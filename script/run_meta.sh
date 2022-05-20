# $1 city  $2 gpu_id

city=$1
gpu_id=$2

bash script/run_meta_single_modal.sh SASRec $city $gpu_id
bash script/run_meta_single_modal.sh txt $city $gpu_id
bash script/run_meta_single_modal.sh img $city $gpu_id
bash script/run_meta_fusion.sh $city $gpu_id
