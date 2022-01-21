NITER=3
(
for K in 10 20
do
    for T in 30
    do
        for N in 50 30 10
        do
            for d in 10 20 30
            do
                for shI in 10 100 300
                do
                    for shU in 10 100 300
                    do
                	((i=i%NITER)); ((i++==0)) && wait
                        python run_fullcmp.py -d $d -K $K -T $T -N $N --shU 10 --shI 50& 
                    done
                done
            done
        done
    done
done
)
