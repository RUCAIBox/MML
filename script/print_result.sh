city_list="shanghai hangzhou changsha"
echo "          R@5    R@10   R@20    M@5    M@10   M@20    N@5    N@10   N@20"
for city in $city_list
do
    printf "%-10s" $city
    tail output/$city/exp/up/down/attention-fusion.meta-test.IEG.log -n 2 | head -n 1 | sed -E 's/\)|\]|\}//g' | awk -F',' '{ printf("%.4f %.4f %.4f  %.4f %.4f %.4f  %.4f %.4f %.4f\n", $2, $4, $6, $8, $10, $12, $14, $16, $18) }'
done
