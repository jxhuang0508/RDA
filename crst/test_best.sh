
for i in {30000..10000000..200}
do
   echo "TEST $i MODEL"
   python evaluate_advent_best.py --test-flipping --data-dir ../RDA/data/Cityscapes --restore-from ../RDA/experiments/snapshots/GTA2Cityscapes_DeepLabv2_RDA/model_$i.pth --save ../RDA/experiments/GTA2Cityscapes_RDA
done
