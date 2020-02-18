for VAR in {0..11}
do
    rm -rf data$VAR
    cp -r ../simulator/base_mesh data$VAR
done
