  *      g@)       =2T
Iterator::Root::ParallelMapV2??ʅʧ?!?29@)??ʅʧ?1?29@:Preprocessing2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?V
?\??!?m??;@)???n???1?_^???7@:Preprocessing2E
Iterator::Root?:????!??T?xGG@)ù?O??1? ???p5@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate?PMI????!?a???1@)B???D??1?=??#@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice+mq??d??!??́g? @)+mq??d??1??́g? @:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipmT?YO??!#	???J@)5ӽN?˂?1?0??%?@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*t^c??~?!??<t/@)*t^c??~?1??<t/@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??Hi6???!?m?T?4@)S@?? ku?1?\p???@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JCPU_ONLYb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.