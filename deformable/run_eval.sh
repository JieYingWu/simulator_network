files=('1_0' '1_1' '1_2' '1_3' '1_4' '1_5' '1_6' '2_0' '2_1' '2_2' '2_3' '2_4' '2_5')

for i in "${files[@]}"
do
    echo $i
#    python evaluate.py ../simulator/data$i/ ../../dataset/2019-10-09-test/camera/data$i/
#    python evaluate.py ../../dataset/2019-10-09-test/simulator/data$i/ ../../dataset/2019-10-09-test/camera/data$i/
#    python evaluate_fem.py ../simulator/data$i/ ../../dataset/2019-10-09-test/simulator/data$i/
    python evaluate_fem.py ../simulator/base_mesh/ ../simulator/data$i/
    #../../dataset/2019-10-09-test/simulator/data$i/
done
