#!/bin/bash

# MOD="ae"
# MOD="af"
# MOD="ag"
# MOD="ah"
# MOD="ai"
# MOD="aj"
MOD="ab" # s10
# MOD="ac"

get_seeded_random()
{
    seed="$1";
    openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
	    </dev/zero 2>/dev/null;
}

seed=0;

# DSET="multiview_${MOD}_s6"
DSET="traj_${MOD}_s10"
PREFIX="t${MOD}"

echo "searching for ${DSET}"
cd ${DSET}
echo $PWD
# ls *rd > ../${DSET}.txt
find . -name "*npz" > ../${DSET}.txt
cd ..

cat ${DSET}.txt | grep "CLEVR_new_0000[0-6]" | shuf --random-source=<(get_seeded_random $seed) > ${PREFIX}t.txt
cat ${DSET}.txt | grep -v "CLEVR_new_0000[0-6]" | shuf --random-source=<(get_seeded_random $seed) > ${PREFIX}v.txt

# add the data path as a prefix
sed -i -e "s/^/${DSET}\//" ${PREFIX}t.txt
sed -i -e "s/^/${DSET}\//" ${PREFIX}v.txt
cat ${PREFIX}t.txt ${PREFIX}v.txt > ${PREFIX}a.txt

head -10 ${PREFIX}t.txt > ${PREFIX}ten.txt
head -1 ${PREFIX}t.txt > ${PREFIX}one.txt
wc -l ${PREFIX}a.txt
wc -l ${PREFIX}t.txt
wc -l ${PREFIX}v.txt
wc -l ${PREFIX}ten.txt
wc -l ${PREFIX}one.txt

cat ${PREFIX}t.txt | sort > ${PREFIX}sort.txt

# put a few redundant lines into one.txt, so that we can train with B>1
cat ${PREFIX}one.txt ${PREFIX}one.txt ${PREFIX}one.txt ${PREFIX}one.txt > temp.txt
cat temp.txt > ${PREFIX}one.txt
rm temp.txt
