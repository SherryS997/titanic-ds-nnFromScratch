  *	?A`?Жh@2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?u?|?H??!??o!+@@)<l"3???1?3??S?4@:Preprocessing2T
Iterator::Root::ParallelMapV2}w+K??!ٵ??.&4@)}w+K??1ٵ??.&4@:Preprocessing2E
Iterator::Rootl#??fF??!JrԂE#C@)?7?a?A??1?.>\ 2@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat|,G?@??!???Z96@)??u?ݡ?1???S?1@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice??d????!???݇'@)??d????1???݇'@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zipڏ?a??!??+}??N@)?????m??1p3?%?J@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?A?F????!A4???k@)?A?F????1A4???k@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????????!c??MgA@)=?u??s?1??ٴ??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.