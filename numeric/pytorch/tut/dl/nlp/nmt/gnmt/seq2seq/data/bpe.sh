# Learn Shared BPE
src="en"
tgt="fr"
OUTPUT_DIR="${DL_PATH}/nmt/${src}-${tgt}/data"
for merge_ops in 40000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  # echo "${OUTPUT_DIR}/train.tok.clean.${src}-${tgt}"
    subword-nmt learn-bpe -s $merge_ops --input "${OUTPUT_DIR}/train.tok.clean.${src}-${tgt}" --output  "${OUTPUT_DIR}/bpe.${merge_ops}" --num-workers 8

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in ${src} ${tgt}; do
    for f in "train" "valid" "test"; do
      outfile="${OUTPUT_DIR}/${f}.tok.clean.bpe.${merge_ops}.${lang}"
      subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" --num-workers 8 < "${OUTPUT_DIR}/${f}.tok.clean.${lang}"  > "${outfile}"
      echo ${outfile}
    done
  done
   
#   Create vocabulary file for BPE

   cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.${src}" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.${tgt}" > "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.${src}-${tgt}" 
    subword-nmt get-vocab  -i "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.${src}-${tgt}"   -o  "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

echo "All done."
