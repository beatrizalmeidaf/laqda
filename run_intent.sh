cuda=-1
FreezeLayer=6
sample=100
commont=Bert-PN-addQ-CE-att
k=4

for path in 01 02 03 04 05;
do
    python ./src_org/main.py \
        --dataset Banking77 \
        --dataFile data/BANKING77/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 1 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Clinc150 \
        --dataFile data/OOS/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 1 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Hwu64 \
        --dataFile data/HWU64/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 1 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Liu \
        --dataFile data/Liu/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 1 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Banking77 \
        --dataFile data/BANKING77/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 5 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Clinc150 \
        --dataFile data/OOS/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 5 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Hwu64 \
        --dataFile data/Hwu64/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 5 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont

    python ./src_org/main.py \
        --dataset Liu \
        --dataFile data/Liu/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result2 \
        --numKShot 5 \
        --sample=$sample \
        --numDevice=$cuda \
        --numQShot 5 \
        --k=$k \
        --numFreeze=$FreezeLayer \
        --commont=$commont
    
done
