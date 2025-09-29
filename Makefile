train-zero1:
	deepspeed --master_port=29500 code/lesson3/zero1.py 2>&1 | tee training.log
