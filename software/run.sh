#!/bin/sh

ls main_STC2.m | (while
read line;
do
	echo "Start to run the matlab file:"${line}
	nohup ./matlab.ln <$line> $(date '+%Y-%m-%d_%H-%M-%S')${line}.out&
	touch $!".pid"
done)
