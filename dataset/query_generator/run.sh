#!/bin/bash

echo "v-8"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 8 --num_queries 200 --anchor vertex --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "v-16"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 16 --num_queries 200 --anchor vertex --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "v-24"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 24 --num_queries 200 --anchor vertex --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "v-32"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 32 --num_queries 200 --anchor vertex --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "v-40"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 40 --num_queries 200 --anchor vertex --output ../../../dataset/enumeration/patents/ --pos_neg neg

echo "e-8"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 8 --num_queries 200 --anchor edge --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "e-16"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 16 --num_queries 200 --anchor edge --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "e-24"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 24 --num_queries 200 --anchor edge --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "e-32"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 32 --num_queries 200 --anchor edge --output ../../../dataset/enumeration/patents/ --pos_neg neg
echo "e-40"
python generate_queries.py --input ../../../../raw_dataset/Dataset-In-mem/patents/data_graph/patents.graph --query_size 40 --num_queries 200 --anchor edge --output ../../../dataset/enumeration/patents/ --pos_neg neg

