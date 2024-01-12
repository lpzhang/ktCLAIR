#!/bin/bash

# challenge=MultiCoil/SingleCoil
challenge=MultiCoil
# task=Cine/Mapping
task=Mapping
# subset=train/val
subset=train

flist=/raid/export/lpzhang/CMRxRecon/datasplitlists/${subset}.lst.seed42.s80s20
src=/raid/export/lpzhang/CMRxRecon/MICCAIChallenge2023/ChallengeData/${challenge}/${task}/TrainingSet/FullSample
dst=/raid/export/lpzhang/CMRxRecon/MICCAIChallenge2023/ChallengeData/${challenge}/${task}/TrainingSet/${subset}
mkdir ${dst}

echo --challenge=${challenge}
echo --task=${task}
echo --subset=${subset}
echo --flist=${flist}
echo --src=${src}
echo --dst=${dst}

for f in $(cat ${flist});
do
        echo $f
        ln -s ${src}/${f} ${dst}
done
