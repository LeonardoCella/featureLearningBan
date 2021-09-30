NITER=3
(
for K in 10 20
do
    for T in 15 30
    do
        for N in 10 30 50
        do
            for d in 10 20 30
            do
        	((i=i%NITER)); ((i++==0)) && wait
                python run.py -d $d -K $K -T $T -N $N --shU 10 --shI 50& 
            done
        done
    done
done
)
