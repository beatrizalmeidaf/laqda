cuda=-1
commont=Bert-PN-addQ-CE-att
for path in 01 02 03 04 05;
do
    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset HuffPost \
        --dataFile data/HuffPost/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset 20News \
        --dataFile data/20News/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset 20News \
        --dataFile data/20News/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset Amazon \
        --dataFile data/Amazon/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 5 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset Amazon \
        --dataFile data/Amazon/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset Reuters \
        --dataFile data/Reuters/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 5 \
        --k 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numQShot 15 \
        --numFreeze 6 \
        --commont=$commont

    python src_org/main.py \
        --dataset Reuters \
        --dataFile data/Reuters/few_shot/${path} \
        --fileVocab="bert-base-uncased" \
        --fileModelConfig="bert-base-uncased" \
        --fileModel="bert-base-uncased" \
        --fileModelSave result${path} \
        --numKShot 1 \
        --sample 1 \
        --numDevice=$cuda \
        --numQShot 15 \
        --numFreeze 6 \
        --commont=$commont
done