NITER=5
(
for K in 10 20
do
    for T in 30 50
    do
        for N in 10 20 30
        do
            for d in 10 20 30
            do
                for shU in 10 20 50
                do
		    for shI in 10 20 50
		    do
			    
                    	((i=i%NITER)); ((i++==0)) && wait
                        python run.py -d $d -K $K -T $T -N $N --shU $shU --shI $shI& 
	            done
                done
            done
        done
    done
done
)
