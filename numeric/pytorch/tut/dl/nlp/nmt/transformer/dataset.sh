#!/bin/bash
CURR_DIR=`pwd`
OUTPUT_DIR="/mnt/dl/Translation/WMT_15/en-fr"
src="en"
tgt="fr"
lang=en-fr
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
  cd ${OUTPUT_DIR}/mosesdecoder
  git reset --hard 8c5eaa1a122236bbf927bde4ec610906fea599e6
  cd -
fi


# Tokenize data
# for lang in ${src} ${tgt}; do
#     for f in ${OUTPUT_DIR}/*.${lang}; do
#         echo "Tokenizing $f..."
#         ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l ${lang} -threads 12 < $f >  ${f%.*}.tok.${lang}
#     done
# done

# Clean all corpora
# for f in ${OUTPUT_DIR}/*.tok.${tgt}; do
#     fbase=${f%.*}
#     echo "Cleaning ${fbase}..."
#     ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase ${tgt} ${src} "${fbase}.clean" 1 256
# done

# Filter datasets
# python3 filter_dataset.py \
#    -f1 ${OUTPUT_DIR}/train.tok.clean.${src} \
#    -f2 ${OUTPUT_DIR}/train.tok.clean.${tgt} \
#    -f1_out ${OUTPUT_DIR}/train.tok.clean.dl.${src} \
#    -f2_out ${OUTPUT_DIR}/train.tok.clean.dl.${tgt} 

# python3 filter_dataset.py \
#    -f1 ${OUTPUT_DIR}/valid.tok.clean.${src} \
#    -f2 ${OUTPUT_DIR}/valid.tok.clean.${tgt} \
#    -f1_out ${OUTPUT_DIR}/valid.tok.clean.dl.${src} \
#    -f2_out ${OUTPUT_DIR}/valid.tok.clean.dl.${tgt} 

# cp ${OUTPUT_DIR}/test.tok.clean.${src} ${OUTPUT_DIR}/test.tok.clean.dl.${src}
# cp ${OUTPUT_DIR}/test.tok.clean.${tgt} ${OUTPUT_DIR}/test.tok.clean.dl.${tgt}

# Learn Shared BPE

# for merge_ops in 40000; do
#   echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
#   # echo "${OUTPUT_DIR}/train.tok.clean.${src}-${tgt}"
#     cat "${OUTPUT_DIR}/train.tok.clean.dl.${src}" "${OUTPUT_DIR}/train.tok.clean.dl.${tgt}" > "${OUTPUT_DIR}/train.tok.clean.dl.${src}-${tgt}" 
#     subword-nmt learn-bpe -s $merge_ops --input "${OUTPUT_DIR}/train.tok.clean.dl.${src}-${tgt}" --output  "${OUTPUT_DIR}/bpe.${merge_ops}" --num-workers 12

#   echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
#   for lang in ${src} ${tgt}; do
#     for f in "train" "valid" "test"; do
#         echo "Apply BPE to ${OUTPUT_DIR}/${f}.tok.clean.dl.${lang}"
#       outfile="${OUTPUT_DIR}/${f}.tok.clean.dl.bpe.${merge_ops}.${lang}"
#       subword-nmt apply-bpe -c "${OUTPUT_DIR}/bpe.${merge_ops}" --num-workers 12 < "${OUTPUT_DIR}/${f}.tok.clean.dl.${lang}"  > "${outfile}"
#     done
#   done
   
# #   Create vocabulary file for BPE
#   echo "Creating vocabulary..."

#    cat "${OUTPUT_DIR}/train.tok.clean.dl.bpe.${merge_ops}.${src}" "${OUTPUT_DIR}/train.tok.clean.dl.bpe.${merge_ops}.${tgt}" > "${OUTPUT_DIR}/train.tok.clean.dl.bpe.${merge_ops}.${src}-${tgt}" 
#     subword-nmt get-vocab  -i "${OUTPUT_DIR}/train.tok.clean.dl.bpe.${merge_ops}.${src}-${tgt}"   -o  "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

# done

# echo "All done."
