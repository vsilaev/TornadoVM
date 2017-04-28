#!/bin/bash

PACKAGE="tornado.benchmarks"
BENCHMARKS="sadd saxpy sgemm spmv addvector dotvector rotatevector rotateimage convolvearray convolveimage"
MAIN_CLASS="Benchmark"

TORNADO_CMD="tornado"
DATE=$(date '+%Y-%m-%d-%H:%M')

RESULTS_ROOT="${TORNADO_ROOT}/var/results"
BM_ROOT="${RESULTS_ROOT}/${DATE}"

if [ ! -d ${BM_ROOT} ]; then
  mkdir -p ${BM_ROOT}
fi

LOGFILE="${BM_ROOT}/bm"

ITERATIONS=10

if [ ! -z "${TORNADO_FLAGS}" ];then
  echo ${TORNADO_FLAGS} > "${BM_ROOT}/tornado.flags"
fi

#echo $(git rev-parse HEAD) > "${BM_ROOT}/git.sha"

for bm in ${BENCHMARKS}; do
	for (( i=0; i<${ITERATIONS}; i++ )); do
		echo "running ${i} ${bm} ..."
		${TORNADO_CMD} ${TORNADO_FLAGS} tornado.benchmarks.BenchmarkRunner ${PACKAGE}.${bm}.${MAIN_CLASS} >> "${LOGFILE}-${bm}-${i}.log"
	done
done

