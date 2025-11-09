SOURCE_DIR="/raid/user_beatrizalmeida/laqda_review_results_br/"

DEST_DIR="~/laqda/results/checkpoints/"


mkdir -p $DEST_DIR

#    -a: Modo "archive" (preserva permissões, etc.)
#    -m: Remove diretórios de origem que ficarem vazios (prune)
#    -v: Verbose 
#    --include='*/': Inclui todos os diretórios para que o rsync possa "entrar" neles
#    --include='acc_best_model.pth': Inclui apenas arquivos com esse nome
#    --exclude='*': Exclui todos os outros arquivos
rsync -amv --include='*/' --include='acc_best_model.pth' --exclude='*' $SOURCE_DIR $DEST_DIR