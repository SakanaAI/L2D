RND_PORT=$(($RANDOM%1000+12000))

echo $RND_PORT

deepspeed --module --include localhost:$1 --master_port $RND_PORT training -c cfgs/training.cfg --extra_config2 $2 "${@:3}"