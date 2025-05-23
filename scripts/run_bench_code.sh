RND_PORT=$(($RANDOM%1000+12000))

echo $RND_PORT

deepspeed --module --include localhost:$1 --master_port $RND_PORT benchmark_code --extra_config1 $2 --micro_batch_size 16 "${@:3}"