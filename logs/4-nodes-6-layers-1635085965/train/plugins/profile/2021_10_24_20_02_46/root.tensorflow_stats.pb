"??
BHostIDLE"IDLE1NbX?`?@ANbX?`?@ai??m ??ii??m ???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1sh???;?@9sh???;?@Ash???;?@Ish???;?@a?#s????i??r?&????Unknown?
iHostWriteSummary"WriteSummary(1??Mb?Q@9??Mb?Q@A??Mb?Q@I??Mb?Q@a??b t?iũ?K????Unknown?
^HostGatherV2"GatherV2(1??(\??C@9??(\??C@A??(\??C@I??(\??C@a-)?_f?i?????????Unknown
vHost_FusedMatMul"sequential_38/dense_128/Relu(1?"??~?B@9?"??~?B@A?"??~?B@I?"??~?B@a???c?5e?i?p(??	???Unknown
cHostDataset"Iterator::Root(1?G?z?Q@9?G?z?Q@A?O??n"B@I?O??n"B@a+?R? xd?i????X???Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1?C?l??J@9?C?l??J@A? ?rhQA@I? ?rhQA@a}?e1?c?iQT?1???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1}?5^?IA@9}?5^?IA@A}?5^?IA@I}?5^?IA@aF??0??c?i?>??hE???Unknown
[	HostPow"
Adam/Pow_1(1??v???@@9??v???@@A??v???@@I??v???@@a7???Ыb?i??ztX???Unknown
?
HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1? ?rh?C@9? ?rh?C@Aw??/m@@Iw??/m@@a?9??b?i?|
?j???Unknown
lHostIteratorGetNext"IteratorGetNext(1?p=
?c?@9?p=
?c?@A?p=
?c?@I?p=
?c?@aQ8`6?a?iM?@V|???Unknown
?HostMatMul".gradient_tape/sequential_38/dense_131/MatMul_1(1+???>@9+???>@A+???>@I+???>@am? ca?inE?`?????Unknown
[HostAddV2"Adam/add(1fffff?<@9fffff?<@Afffff?<@Ifffff?<@a0?
 f=`?iTP???????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_9/ResourceApplyAdam(1\???(?<@9\???(?<@A\???(?<@I\???(?<@a{???7`?i??xe.????Unknown
?HostResourceApplyAdam"%Adam/Adam/update_11/ResourceApplyAdam(1????ҍ<@9????ҍ<@A????ҍ<@I????ҍ<@aV%?w`?i%f?K????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1-??罹:@9-??罹:@A-??罹:@I-??罹:@aW#?^?iѧ?cV????Unknown
eHost
LogicalAnd"
LogicalAnd(1?G?zT:@9?G?zT:@A?G?zT:@I?G?zT:@aD?J?]?i?I??2????Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_2(1bX9??9@9bX9??9@AbX9??9@IbX9??9@a???f?\?i+Y'??????Unknown
?HostResourceApplyAdam"%Adam/Adam/update_13/ResourceApplyAdam(1?t?8@9?t?8@A?t?8@I?t?8@a?&?+?/[?i?B=?E????Unknown
`HostDivNoNan"
div_no_nan(1????̬7@9????̬7@A????̬7@I????̬7@aw?Z?iDM?8????Unknown
nHostCast"sequential_38/dense_128/Cast(1??/?d7@9??/?d7@A??/?d7@I??/?d7@a??
?gZ?i??-????Unknown
?HostMatMul",gradient_tape/sequential_38/dense_130/MatMul(1Zd;?OM6@9Zd;?OM6@AZd;?OM6@IZd;?OM6@a]??i],Y?i?˸[l???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_131/MatMul(1??|?5>6@9??|?5>6@A??|?5>6@I??|?5>6@a7??PY?i?5?+???Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1????ƫ5@9????ƫ5@A????ƫ5@I????ƫ5@a4?vX?i??E58???Unknown
tHostReadVariableOp"Adam/Cast/ReadVariableOp(11?Z?5@91?Z?5@A1?Z?5@I1?Z?5@a??9??mX?i????kD???Unknown
?HostReadVariableOp".sequential_38/dense_131/BiasAdd/ReadVariableOp(1B`??"{5@9B`??"{5@AB`??"{5@IB`??"{5@a ?'!?X?i??Kl?P???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_133/MatMul(1?(\?µ4@9?(\?µ4@A?(\?µ4@I?(\?µ4@a?|t?W`W?i???;\???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1??? ?24@9??? ?24@A??? ?24@I??? ?24@aB?i!e?V?i?t?ʡg???Unknown
?HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1?E???43@9?E???43@A?E???43@I?E???43@a??_	?U?i?CU?xr???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_134/MatMul(1?S㥛?1@9?S㥛?1@A?S㥛?1@I?S㥛?1@a??N??S?i?j??m|???Unknown
?HostResourceApplyAdam"%Adam/Adam/update_12/ResourceApplyAdam(1?Zd?0@9?Zd?0@A?Zd?0@I?Zd?0@a?8,???R?i?/K߅???Unknown
g HostStridedSlice"strided_slice(1`??"?0@9`??"?0@A`??"?0@I`??"?0@aՃ۟?,R?i?n???????Unknown
?!HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      0@9      0@A      0@I      0@a???LXR?iB?%;?????Unknown
`"HostGatherV2"
GatherV2_1(1
ףp=?/@9
ףp=?/@A
ףp=?/@I
ףp=?/@a3?c??Q?i͙W??????Unknown
?#HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?5^?I?.@9?5^?I?.@A?5^?I?.@I?5^?I?.@a:??6?aQ?i?s??????Unknown
?$HostReadVariableOp".sequential_38/dense_130/BiasAdd/ReadVariableOp(1?~j?t?-@9?~j?t?-@A?~j?t?-@I?~j?t?-@a<??d ?P?iz???????Unknown
?%HostMatMul",gradient_tape/sequential_38/dense_129/MatMul(1?MbX?,@9?MbX?,@A?MbX?,@I?MbX?,@aZ??]&ZP?i?E?&????Unknown
?&HostBiasAddGrad"9gradient_tape/sequential_38/dense_134/BiasAdd/BiasAddGrad(1V-?,@9V-?,@AV-?,@IV-?,@a?R??1P?i/?$3????Unknown
?'HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?&1?,@9?&1?,@A?&1?,@I?&1?,@a?k??7?O?iܽ?!????Unknown
?(HostReadVariableOp".sequential_38/dense_134/BiasAdd/ReadVariableOp(1?G?z?)@9?G?z?)@A?G?z?)@I?G?z?)@a???V?DM?i? s????Unknown
?)HostMatMul".gradient_tape/sequential_38/dense_133/MatMul_1(1j?t??(@9j?t??(@Aj?t??(@Ij?t??(@a+?zL?iЀ??u????Unknown
?*HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1?"??~?'@9?"??~?'@A?"??~?'@I?"??~?'@a=\K??J?i?F??#????Unknown
?+HostBiasAddGrad"9gradient_tape/sequential_38/dense_133/BiasAdd/BiasAddGrad(1/?$?'@9/?$?'@A/?$?'@I/?$?'@a.?S???J?i??N??????Unknown
?,HostResourceApplyAdam"$Adam/Adam/update_8/ResourceApplyAdam(1NbX9?&@9NbX9?&@ANbX9?&@INbX9?&@ar9?Ć?I?i????-????Unknown
?-HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1?/?$?&@9?/?$?&@A?/?$?&@I?/?$?&@a?t???lI?i?͈????Unknown
v.HostSub"%binary_crossentropy/logistic_loss/sub(1?MbX9&@9?MbX9&@A?MbX9&@I?MbX9&@ah-?z?I?i[??B?????Unknown
?/HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1?Zd;%@9?Zd;%@A?Zd;%@I?Zd;%@aok?d?G?i"??????Unknown
?0HostMatMul".gradient_tape/sequential_38/dense_129/MatMul_1(1???(\%@9???(\%@A???(\%@I???(\%@a6кMz?G?i?}~z????Unknown
?1HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1T㥛??#@9T㥛??#@AT㥛??#@IT㥛??#@a?Ú?oF?i?.?uQ
???Unknown
u2HostReadVariableOp"div_no_nan/ReadVariableOp(1`??"??"@9`??"??"@A`??"??"@I`??"??"@aF1?}IkE?i??DH????Unknown
j3HostMean"binary_crossentropy/Mean(1?V?"@9?V?"@A?V?"@I?V?"@a??`o?E?i!n?n????Unknown
?4HostMatMul".gradient_tape/sequential_38/dense_130/MatMul_1(1V-??"@9V-??"@AV-??"@IV-??"@a??!CE?i??a?2???Unknown
V5HostMean"Mean(1?~j?t?"@9?~j?t?"@A?~j?t?"@I?~j?t?"@a??L??D?i?Ig,p???Unknown
?6HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1???MbP"@9???MbP"@A???MbP"@I???MbP"@aVt??D?iզ(,?$???Unknown
?7HostResourceApplyAdam"%Adam/Adam/update_10/ResourceApplyAdam(1X9?ȶ!@9X9?ȶ!@AX9?ȶ!@IX9?ȶ!@a??????C?i???Ӛ)???Unknown
y8Host_FusedMatMul"sequential_38/dense_134/BiasAdd(1?Zd;? @9?Zd;? @A?Zd;? @I?Zd;? @a,'5QC?i+0?].???Unknown
?9Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1fffff? @9fffff? @Afffff? @Ifffff? @a??+?B?iq??r3???Unknown
?:HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a?JC?h?B?iD??7???Unknown
?;HostBiasAddGrad"9gradient_tape/sequential_38/dense_132/BiasAdd/BiasAddGrad(1?Zd; @9?Zd; @A?Zd; @I?Zd; @a???'bRB?i4r??S<???Unknown
\<HostGreater"Greater(1q=
ף?@9q=
ף?@Aq=
ף?@Iq=
ף?@a?E??B?in?P?@???Unknown
V=HostSum"Sum_2(133333?@933333?@A33333?@I33333?@a??pE ?A?i6??PNE???Unknown
?>HostReadVariableOp"-sequential_38/dense_132/MatMul/ReadVariableOp(1     ?@9     ?@A     ?@I     ?@a?3?(?6A?iC	??I???Unknown
w?HostDataset""Iterator::Root::ParallelMapV2::Zip(1^?I?Z@9^?I?Z@A?t?V@I?t?V@an|?T?A?i?????M???Unknown
?@HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1?p=
?#@9?p=
?#@A?p=
?#@I?p=
?#@aOo:??A?i>??[$R???Unknown
~AHostSelect"*binary_crossentropy/logistic_loss/Select_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@a??J(\?@?i???2\V???Unknown
?BHostReadVariableOp"-sequential_38/dense_133/MatMul/ReadVariableOp(1V-??@9V-??@AV-??@IV-??@ag?a???@?is????Z???Unknown
?CHostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1o???!@9o???!@Ao???!@Io???!@a??????iNݻ?^???Unknown
tDHostAddV2"!binary_crossentropy/logistic_loss(1?Zd?@9?Zd?@A?Zd?@I?Zd?@a?wΞiM??i???kb???Unknown
?EHostReadVariableOp".sequential_38/dense_128/BiasAdd/ReadVariableOp(11?Z?@91?Z?@A1?Z?@I1?Z?@a|?XݯZ>?i9b?7f???Unknown
vFHost_FusedMatMul"sequential_38/dense_129/Relu(1?ʡE?s@9?ʡE?s@A?ʡE?s@I?ʡE?s@aɬ3???=?i??[??i???Unknown
?GHostReadVariableOp".sequential_38/dense_133/BiasAdd/ReadVariableOp(1?x?&1?@9?x?&1?@A?x?&1?@I?x?&1?@a
k???<?i<ط?m???Unknown
vHHost_FusedMatMul"sequential_38/dense_131/Relu(1+????@9+????@A+????@I+????@a圎к.<?i2?q???Unknown
vIHostNeg"%binary_crossentropy/logistic_loss/Neg(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a????S;?ig5?|t???Unknown
?JHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1J+?@9J+?@AJ+?@IJ+?@a~1Dr0;?i??S?w???Unknown
~KHostMaximum")gradient_tape/binary_crossentropy/Maximum(1+????@9+????@A+????@I+????@a{??K?;?i?v??D{???Unknown
?LHostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1X9???@9X9???@AX9???@IX9???@a??f?;?iS?M?~???Unknown
vMHost_FusedMatMul"sequential_38/dense_130/Relu(1???Mb?@9???Mb?@A???Mb?@I???Mb?@a?9???:?i}%?n?????Unknown
YNHostPow"Adam/Pow(1?MbX?@9?MbX?@A?MbX?@I?MbX?@a??=N?9?i2?q8.????Unknown
jOHostCast"binary_crossentropy/Cast(1w??/]@9w??/]@Aw??/]@Iw??/]@a???G>9?iu}jV????Unknown
~PHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?$???@9?$???@A?$???@I?$???@a??)???8?i??_h????Unknown
tQHostSigmoid"sequential_38/dense_134/Sigmoid(1?p=
ף@9?p=
ף@A?p=
ף@I?p=
ף@a???m8?i?۽?u????Unknown
?RHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@aiĠ=H38?iߏ?)|????Unknown
vSHost_FusedMatMul"sequential_38/dense_133/Relu(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a??x?}?7?i??vY{????Unknown
?THostMatMul",gradient_tape/sequential_38/dense_132/MatMul(1?Q???@9?Q???@A?Q???@I?Q???@a	???7?i.7?n????Unknown
vUHost_FusedMatMul"sequential_38/dense_132/Relu(1?Q???@9?Q???@A?Q???@I?Q???@a	???7?ijs??b????Unknown
tVHostAssignAddVariableOp"AssignAddVariableOp(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@a\5(!??7?iq??WS????Unknown
?WHostMatMul".gradient_tape/sequential_38/dense_134/MatMul_1(1???(\?@9???(\?@A???(\?@I???(\?@aYV??47?i<??9????Unknown
?XHostReluGrad".gradient_tape/sequential_38/dense_129/ReluGrad(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@a?U *6?i?m?7?????Unknown
?YHostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1`??"?YM@9`??"?YM@AV-??o@IV-??o@a%?,O5?5?ivS?>?????Unknown
?ZHostMatMul",gradient_tape/sequential_38/dense_128/MatMul(1V-2@9V-2@AV-2@IV-2@a!?/?۪5?ip??r????Unknown
z[HostLog1p"'binary_crossentropy/logistic_loss/Log1p(1??~j??@9??~j??@A??~j??@I??~j??@a?2?e5?i??MJ????Unknown
]\HostCast"Adam/Cast_1(1j?t??@9j?t??@Aj?t??@Ij?t??@ag??]?4?i{??????Unknown
?]HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a7??!?4?i???S????Unknown
?^HostMatMul".gradient_tape/sequential_38/dense_132/MatMul_1(1??"???@9??"???@A??"???@I??"???@a?>qP4?i? ?ݲ???Unknown
?_HostReadVariableOp".sequential_38/dense_132/BiasAdd/ReadVariableOp(1?"??~?@9?"??~?@A?"??~?@I?"??~?@a[???84?i8?_=d????Unknown
?`HostBiasAddGrad"9gradient_tape/sequential_38/dense_130/BiasAdd/BiasAddGrad(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a?H?c,?3?iaq?"׷???Unknown
?aHostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a??e??~3?i??F????Unknown
?bHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1d;?O??@9d;?O??@Ad;?O??@Id;?O??@a?JC?h?2?i{?L?????Unknown
?cHostBiasAddGrad"9gradient_tape/sequential_38/dense_129/BiasAdd/BiasAddGrad(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@avnq	?i2?i?4?|?????Unknown
vdHostAssignAddVariableOp"AssignAddVariableOp_4(1+??N@9+??N@A+??N@I+??N@aY?$Xh2?i??ه8????Unknown
veHostMul"%binary_crossentropy/logistic_loss/mul(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a???]2?i'??E?????Unknown
?fHostReadVariableOp"-sequential_38/dense_128/MatMul/ReadVariableOp(1#??~j?@9#??~j?@A#??~j?@I#??~j?@aS#j?3?1?ikGvl?????Unknown
XgHostCast"Cast_3(1/?$??@9/?$??@A/?$??@I/?$??@a???=?1?i?0??????Unknown
?hHostReadVariableOp"-sequential_38/dense_130/MatMul/ReadVariableOp(1??C?l@9??C?l@A??C?l@I??C?l@ai????1?iٺ-X3????Unknown
viHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??K7?A@9??K7?A@A??K7?A@I??K7?A@aNm&ڣ1?i}?r?g????Unknown
ejHostAddN"Adam/gradients/AddN(1??K7?A@9??K7?A@A??K7?A@I??K7?A@aNm&ڣ1?i!V?N?????Unknown
?kHostReluGrad".gradient_tape/sequential_38/dense_130/ReluGrad(1V-?@9V-?@AV-?@IV-?@a?R??10?ik?8??????Unknown
blHostDivNoNan"div_no_nan_1(1??|?5^@9??|?5^@A??|?5^@I??|?5^@a???c?0?iDE??????Unknown
omHostReadVariableOp"Adam/ReadVariableOp(1? ?rh?@9? ?rh?@A? ?rh?@I? ?rh?@a???/?i?????????Unknown
?nHostReadVariableOp"-sequential_38/dense_129/MatMul/ReadVariableOp(15^?I@95^?I@A5^?I@I5^?I@aP??4|.?i?\??|????Unknown
?oHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1?I+?
@9?I+?
@A?I+?
@I?I+?
@aD????-?i?l	?[????Unknown
XpHostCast"Cast_5(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a???:??+?im-z????Unknown
XqHostEqual"Equal(1?I+?@9?I+?@A?I+?@I?I+?@a1giw??+?i??s?????Unknown
?rHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a????(?+?i?N!֍????Unknown
?sHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1ffffff@9ffffff@Affffff@Iffffff@a??]V?i*?i??fp4????Unknown
?tHostBiasAddGrad"9gradient_tape/sequential_38/dense_128/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@aͬl?)?id??^?????Unknown
?uHostBiasAddGrad"9gradient_tape/sequential_38/dense_131/BiasAdd/BiasAddGrad(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a?Ŗl?(?i??q?P????Unknown
?vHostReluGrad".gradient_tape/sequential_38/dense_128/ReluGrad(1???Q?@9???Q?@A???Q?@I???Q?@aàr?0?(?i?.??????Unknown
?wHostReluGrad".gradient_tape/sequential_38/dense_133/ReluGrad(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a??u.?>(?iH??\????Unknown
?xHostReadVariableOp".sequential_38/dense_129/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@a??L?'(?i??1?????Unknown
vyHostExp"%binary_crossentropy/logistic_loss/Exp(1??Q??@9??Q??@A??Q??@I??Q??@a????p)'?i???Q????Unknown
|zHostSelect"(binary_crossentropy/logistic_loss/Select(1?(\???@9?(\???@A?(\???@I?(\???@a???n??&?i?x?B?????Unknown
V{HostCast"Cast(1?n???@9?n???@A?n???@I?n???@a?E??W&?i`?ų????Unknown
?|HostReadVariableOp"-sequential_38/dense_134/MatMul/ReadVariableOp(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a?????%?i˓?,{????Unknown
?}HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1?G?z@9?G?z@A?G?z@I?G?z@a?^??V?%?i]Y??????Unknown
?~HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@a?X??C%?ig?*(????Unknown
?HostReadVariableOp"-sequential_38/dense_131/MatMul/ReadVariableOp(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@a?X??C%?i?O?A|????Unknown
w?HostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a?n1}?Z$?i?"$??????Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_1(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a?JC?h?"?i???????Unknown
?HostRealDiv")gradient_tape/binary_crossentropy/truediv(1?$??C??9?$??C??A?$??C??I?$??C??a?2O?!?i???????Unknown
~?HostDivNoNan"'binary_crossentropy/weighted_loss/value(1m???????9m???????Am???????Im???????a? X?? ?i?? 0????Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_3(1?I+???9?I+???A?I+???I?I+???am?FE? ?i????????Unknown
U?HostMul"Mul(1??C?l???9??C?l???A??C?l???I??C?l???a3??i8lm?????Unknown
y?HostCast"&gradient_tape/binary_crossentropy/Cast(1? ?rh???9? ?rh???A? ?rh???I? ?rh???a????i !??????Unknown
??HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a???:???i??.??????Unknown
??Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1?O??n??9?O??n??A?O??n??I?O??n??aW???
?i}&??????Unknown
b?HostIdentity"Identity(1!?rh????9!?rh????A!?rh????I!?rh????a??'?@??i???????Unknown?
??HostReluGrad".gradient_tape/sequential_38/dense_131/ReluGrad(1?ʡE????9?ʡE????A?ʡE????I?ʡE????a?{??:??iM?=P????Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1;?O??n??9;?O??n??A;?O??n??I;?O??n??aL΁???i&y???????Unknown
??HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1bX9????9bX9????AbX9????IbX9????a?,Rf?_?i?????????Unknown
??HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1m???????9m???????Am???????Im???????a? X???ix??S????Unknown
x?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1w??/???9w??/???Aw??/???Iw??/???a?^?AJ?ii!???????Unknown
z?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a?e???u?iG?$}????Unknown
??HostReluGrad".gradient_tape/sequential_38/dense_132/ReluGrad(1D?l?????9D?l?????AD?l?????ID?l?????a[.9b???i,Q?\|????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1?Zd;???9?Zd;???A?Zd;???I?Zd;???aj?o?1n?i?^M?????Unknown
??HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1????x???9????x???A????x???I????x???aV?{/?X?i?zx+????Unknown
Y?HostCast"Cast_4(1??ʡE??9??ʡE??A??ʡE??I??ʡE??a?t."ܟ?i????}????Unknown
x?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a,iM?A ?i?R???????Unknown
??HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1????????9????????A????????I????????a,iM?A ?i      ???Unknown*ʒ
uHostFlushSummaryWriter"FlushSummaryWriter(1sh???;?@9sh???;?@Ash???;?@Ish???;?@a?^4N??i?^4N???Unknown?
iHostWriteSummary"WriteSummary(1??Mb?Q@9??Mb?Q@A??Mb?Q@I??Mb?Q@a.+?ؑ?i??%??????Unknown?
^HostGatherV2"GatherV2(1??(\??C@9??(\??C@A??(\??C@I??(\??C@a?r?g???io???,???Unknown
vHost_FusedMatMul"sequential_38/dense_128/Relu(1?"??~?B@9?"??~?B@A?"??~?B@I?"??~?B@a*>??iۂ?i??8?w???Unknown
cHostDataset"Iterator::Root(1?G?z?Q@9?G?z?Q@A?O??n"B@I?O??n"B@a???2??iM<A?????Unknown
?HostDataset">Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate(1?C?l??J@9?C?l??J@A? ?rhQA@I? ?rhQA@a?????`??i,?@C???Unknown
rHostDataset"Iterator::Root::ParallelMapV2(1}?5^?IA@9}?5^?IA@A}?5^?IA@I}?5^?IA@a\?JY??i?(k?K???Unknown
[HostPow"
Adam/Pow_1(1??v???@@9??v???@@A??v???@@I??v???@@a[ZM????iG?{????Unknown
?	HostDataset"4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat(1? ?rh?C@9? ?rh?C@Aw??/m@@Iw??/m@@a?Z`??{??io?N_?????Unknown
l
HostIteratorGetNext"IteratorGetNext(1?p=
?c?@9?p=
?c?@A?p=
?c?@I?p=
?c?@a|M????i
f?????Unknown
?HostMatMul".gradient_tape/sequential_38/dense_131/MatMul_1(1+???>@9+???>@A+???>@I+???>@a??u??~?i?Q??L???Unknown
[HostAddV2"Adam/add(1fffff?<@9fffff?<@Afffff?<@Ifffff?<@a???8N?|?i???Y?????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_9/ResourceApplyAdam(1\???(?<@9\???(?<@A\???(?<@I\???(?<@a?t1??|?i?7?h@????Unknown
?HostResourceApplyAdam"%Adam/Adam/update_11/ResourceApplyAdam(1????ҍ<@9????ҍ<@A????ҍ<@I????ҍ<@aTQ?ԇ?|?i@$?x?????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_2(1-??罹:@9-??罹:@A-??罹:@I-??罹:@aŧ~<??z?i?!?/???Unknown
eHost
LogicalAnd"
LogicalAnd(1?G?zT:@9?G?zT:@A?G?zT:@I?G?zT:@aQX?2/lz?iA?k#?c???Unknown?
vHostAssignAddVariableOp"AssignAddVariableOp_2(1bX9??9@9bX9??9@AbX9??9@IbX9??9@aO?{?Ϳy?iv?6?d????Unknown
?HostResourceApplyAdam"%Adam/Adam/update_13/ResourceApplyAdam(1?t?8@9?t?8@A?t?8@I?t?8@a!??<?+x?i??%?????Unknown
`HostDivNoNan"
div_no_nan(1????̬7@9????̬7@A????̬7@I????̬7@aeu7?w?i??`@????Unknown
nHostCast"sequential_38/dense_128/Cast(1??/?d7@9??/?d7@A??/?d7@I??/?d7@aJ????yw?i?94&???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_130/MatMul(1Zd;?OM6@9Zd;?OM6@AZd;?OM6@IZd;?OM6@a????cav?i???R???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_131/MatMul(1??|?5>6@9??|?5>6@A??|?5>6@I??|?5>6@a0?;Rv?iD??x????Unknown
?HostSelect"8gradient_tape/binary_crossentropy/logistic_loss/Select_3(1????ƫ5@9????ƫ5@A????ƫ5@I????ƫ5@aM?5I?u?i?",????Unknown
tHostReadVariableOp"Adam/Cast/ReadVariableOp(11?Z?5@91?Z?5@A1?Z?5@I1?Z?5@a3???շu?i??!??????Unknown
?HostReadVariableOp".sequential_38/dense_131/BiasAdd/ReadVariableOp(1B`??"{5@9B`??"{5@AB`??"{5@IB`??"{5@a%"??y?u?i??I?????Unknown
?HostMatMul",gradient_tape/sequential_38/dense_133/MatMul(1?(\?µ4@9?(\?µ4@A?(\?µ4@I?(\?µ4@aD?Πg?t?iL7?y7+???Unknown
vHostSum"%binary_crossentropy/weighted_loss/Sum(1??? ?24@9??? ?24@A??? ?24@I??? ?24@a?`?1?Dt?i??7?S???Unknown
?HostDataset"NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[1]::Concatenate[0]::TensorSlice(1?E???43@9?E???43@A?E???43@I?E???43@a2?	?HFs?i??,?Mz???Unknown
?HostMatMul",gradient_tape/sequential_38/dense_134/MatMul(1?S㥛?1@9?S㥛?1@A?S㥛?1@I?S㥛?1@a?9?}?q?i"?Ķ????Unknown
?HostResourceApplyAdam"%Adam/Adam/update_12/ResourceApplyAdam(1?Zd?0@9?Zd?0@A?Zd?0@I?Zd?0@aם??t?p?i^=?K????Unknown
gHostStridedSlice"strided_slice(1`??"?0@9`??"?0@A`??"?0@I`??"?0@a/Z??Y(p?i?b?????Unknown
? HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1      0@9      0@A      0@I      0@a?+??gp?ii?1?????Unknown
`!HostGatherV2"
GatherV2_1(1
ףp=?/@9
ףp=?/@A
ףp=?/@I
ףp=?/@aJX*???o?i????_???Unknown
?"HostResourceApplyAdam"$Adam/Adam/update_5/ResourceApplyAdam(1?5^?I?.@9?5^?I?.@A?5^?I?.@I?5^?I?.@a?W??n?i?Cm?G>???Unknown
?#HostReadVariableOp".sequential_38/dense_130/BiasAdd/ReadVariableOp(1?~j?t?-@9?~j?t?-@A?~j?t?-@I?~j?t?-@a??y*?m?iĽ???[???Unknown
?$HostMatMul",gradient_tape/sequential_38/dense_129/MatMul(1?MbX?,@9?MbX?,@A?MbX?,@I?MbX?,@a????mm?i??WZ	y???Unknown
?%HostBiasAddGrad"9gradient_tape/sequential_38/dense_134/BiasAdd/BiasAddGrad(1V-?,@9V-?,@AV-?,@IV-?,@a?v???l?iz]Օ???Unknown
?&HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?&1?,@9?&1?,@A?&1?,@I?&1?,@a+)?5l?i'?X????Unknown
?'HostReadVariableOp".sequential_38/dense_134/BiasAdd/ReadVariableOp(1?G?z?)@9?G?z?)@A?G?z?)@I?G?z?)@aӐ8?lj?i?۬?????Unknown
?(HostMatMul".gradient_tape/sequential_38/dense_133/MatMul_1(1j?t??(@9j?t??(@Aj?t??(@Ij?t??(@aĶD???h?io n??????Unknown
?)HostResourceApplyAdam"$Adam/Adam/update_6/ResourceApplyAdam(1?"??~?'@9?"??~?'@A?"??~?'@I?"??~?'@aw?QͿg?i?????????Unknown
?*HostBiasAddGrad"9gradient_tape/sequential_38/dense_133/BiasAdd/BiasAddGrad(1/?$?'@9/?$?'@A/?$?'@I/?$?'@a?:e&/?g?i???U???Unknown
?+HostResourceApplyAdam"$Adam/Adam/update_8/ResourceApplyAdam(1NbX9?&@9NbX9?&@ANbX9?&@INbX9?&@aū????f?is?\+???Unknown
?,HostDynamicStitch"/gradient_tape/binary_crossentropy/DynamicStitch(1?/?$?&@9?/?$?&@A?/?$?&@I?/?$?&@a3?l?f?ix?ȸA???Unknown
v-HostSub"%binary_crossentropy/logistic_loss/sub(1?MbX9&@9?MbX9&@A?MbX9&@I?MbX9&@a???ZMf?ih??"X???Unknown
?.HostTile"6gradient_tape/binary_crossentropy/weighted_loss/Tile_1(1?Zd;%@9?Zd;%@A?Zd;%@I?Zd;%@a?Qj?2e?i?\b8m???Unknown
?/HostMatMul".gradient_tape/sequential_38/dense_129/MatMul_1(1???(\%@9???(\%@A???(\%@I???(\%@a? ??Q"e?iۿ@?Z????Unknown
?0HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1T㥛??#@9T㥛??#@AT㥛??#@IT㥛??#@a4?y???c?ii9-^M????Unknown
u1HostReadVariableOp"div_no_nan/ReadVariableOp(1`??"??"@9`??"??"@A`??"??"@I`??"??"@aT???
c?i?1?NX????Unknown
j2HostMean"binary_crossentropy/Mean(1?V?"@9?V?"@A?V?"@I?V?"@ai>??޽b?i?#^-????Unknown
?3HostMatMul".gradient_tape/sequential_38/dense_130/MatMul_1(1V-??"@9V-??"@AV-??"@IV-??"@a???t?b?i??????Unknown
V4HostMean"Mean(1?~j?t?"@9?~j?t?"@A?~j?t?"@I?~j?t?"@a??E?-?b?i?Z??h????Unknown
?5HostResourceApplyAdam"$Adam/Adam/update_4/ResourceApplyAdam(1???MbP"@9???MbP"@A???MbP"@I???MbP"@a??{)?`b?i??$??????Unknown
?6HostResourceApplyAdam"%Adam/Adam/update_10/ResourceApplyAdam(1X9?ȶ!@9X9?ȶ!@AX9?ȶ!@IX9?ȶ!@a??oG??a?iTFlj????Unknown
y7Host_FusedMatMul"sequential_38/dense_134/BiasAdd(1?Zd;? @9?Zd;? @A?Zd;? @I?Zd;? @a???k?`?i6U?~???Unknown
?8Host	ZerosLike":gradient_tape/binary_crossentropy/logistic_loss/zeros_like(1fffff? @9fffff? @Afffff? @Ifffff? @aӯ??c?`?i?:4'???Unknown
?9HostResourceApplyAdam"$Adam/Adam/update_7/ResourceApplyAdam(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a?q?L}?`?i@Gb??7???Unknown
?:HostBiasAddGrad"9gradient_tape/sequential_38/dense_132/BiasAdd/BiasAddGrad(1?Zd; @9?Zd; @A?Zd; @I?Zd; @az$8^J`?id??$H???Unknown
\;HostGreater"Greater(1q=
ף?@9q=
ף?@Aq=
ף?@Iq=
ף?@a??n??`?i??Mk+X???Unknown
V<HostSum"Sum_2(133333?@933333?@A33333?@I33333?@a?A?&??_?iZ?Ih???Unknown
?=HostReadVariableOp"-sequential_38/dense_132/MatMul/ReadVariableOp(1     ?@9     ?@A     ?@I     ?@a7??bu?^?i??aw???Unknown
w>HostDataset""Iterator::Root::ParallelMapV2::Zip(1^?I?Z@9^?I?Z@A?t?V@I?t?V@a.4?Sq^?iTj??????Unknown
??HostSelect"6gradient_tape/binary_crossentropy/logistic_loss/Select(1?p=
?#@9?p=
?#@A?p=
?#@I?p=
?#@a??t?>^?ib?$+?????Unknown
~@HostSelect"*binary_crossentropy/logistic_loss/Select_1(1L7?A`?@9L7?A`?@AL7?A`?@IL7?A`?@aX??oJ ^?iI?\P?????Unknown
?AHostReadVariableOp"-sequential_38/dense_133/MatMul/ReadVariableOp(1V-??@9V-??@AV-??@IV-??@a?t%?[?]?i.F~?????Unknown
?BHostDataset"@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor(1o???!@9o???!@Ao???!@Io???!@aY?~a;\?im?v?????Unknown
tCHostAddV2"!binary_crossentropy/logistic_loss(1?Zd?@9?Zd?@A?Zd?@I?Zd?@a???[?[?isS\;?????Unknown
?DHostReadVariableOp".sequential_38/dense_128/BiasAdd/ReadVariableOp(11?Z?@91?Z?@A1?Z?@I1?Z?@a?????Z?i??V?????Unknown
vEHost_FusedMatMul"sequential_38/dense_129/Relu(1?ʡE?s@9?ʡE?s@A?ʡE?s@I?ʡE?s@a??\???Z?iT_?Fa????Unknown
?FHostReadVariableOp".sequential_38/dense_133/BiasAdd/ReadVariableOp(1?x?&1?@9?x?&1?@A?x?&1?@I?x?&1?@a????-?Y?i?Yx?0????Unknown
vGHost_FusedMatMul"sequential_38/dense_131/Relu(1+????@9+????@A+????@I+????@a!?CIY?i?4????Unknown
vHHostNeg"%binary_crossentropy/logistic_loss/Neg(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a~
??
KX?iʲy?????Unknown
?IHostSum"7gradient_tape/binary_crossentropy/logistic_loss/sub/Sum(1J+?@9J+?@AJ+?@IJ+?@as???6,X?i?/ܢ????Unknown
~JHostMaximum")gradient_tape/binary_crossentropy/Maximum(1+????@9+????@A+????@I+????@af,??bX?i.?AT?'???Unknown
?KHostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1X9???@9X9???@AX9???@IX9???@a?9?[X?i˷? 4???Unknown
vLHost_FusedMatMul"sequential_38/dense_130/Relu(1???Mb?@9???Mb?@A???Mb?@I???Mb?@aFrf#??W?ik?N?????Unknown
YMHostPow"Adam/Pow(1?MbX?@9?MbX?@A?MbX?@I?MbX?@a?hUM??V?i?W5:K???Unknown
jNHostCast"binary_crossentropy/Cast(1w??/]@9w??/]@Aw??/]@Iw??/]@a?N_QqV?iƼ?rV???Unknown
~OHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1?$???@9?$???@A?$???@I?$???@a??A}-?U?i?]?t^a???Unknown
tPHostSigmoid"sequential_38/dense_134/Sigmoid(1?p=
ף@9?p=
ף@A?p=
ף@I?p=
ף@a??arR?U?i???:l???Unknown
?QHostDivNoNan"@gradient_tape/binary_crossentropy/weighted_loss/value/div_no_nan(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a#+'??U?i???v???Unknown
vRHost_FusedMatMul"sequential_38/dense_133/Relu(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@acǮۏPU?i??^?????Unknown
?SHostMatMul",gradient_tape/sequential_38/dense_132/MatMul(1?Q???@9?Q???@A?Q???@I?Q???@a??R?Z?T?iyˋ#????Unknown
vTHost_FusedMatMul"sequential_38/dense_132/Relu(1?Q???@9?Q???@A?Q???@I?Q???@a??R?Z?T?i?<??????Unknown
tUHostAssignAddVariableOp"AssignAddVariableOp(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@a???D??T?i󺸟????Unknown
?VHostMatMul".gradient_tape/sequential_38/dense_134/MatMul_1(1???(\?@9???(\?@A???(\?@I???(\?@at?K?ޡT?i???h????Unknown
?WHostReluGrad".gradient_tape/sequential_38/dense_129/ReluGrad(1?A`?Т@9?A`?Т@A?A`?Т@I?A`?Т@aʶp~?S?i??D?B????Unknown
?XHostDataset".Iterator::Root::ParallelMapV2::Zip[0]::FlatMap(1`??"?YM@9`??"?YM@AV-??o@IV-??o@aS?$?S?i?9?\????Unknown
?YHostMatMul",gradient_tape/sequential_38/dense_128/MatMul(1V-2@9V-2@AV-2@IV-2@a?t?0uCS?i??o?????Unknown
zZHostLog1p"'binary_crossentropy/logistic_loss/Log1p(1??~j??@9??~j??@A??~j??@I??~j??@aږ?<?S?i???'????Unknown
][HostCast"Adam/Cast_1(1j?t??@9j?t??@Aj?t??@Ij?t??@aUD?ÔR?i??`r????Unknown
?\HostBroadcastTo"-gradient_tape/binary_crossentropy/BroadcastTo(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a?k?Z?kR?i2̹4?????Unknown
?]HostMatMul".gradient_tape/sequential_38/dense_132/MatMul_1(1??"???@9??"???@A??"???@I??"???@ay?l-R?i?p˯????Unknown
?^HostReadVariableOp".sequential_38/dense_132/BiasAdd/ReadVariableOp(1?"??~?@9?"??~?@A?"??~?@I?"??~?@a?)8??Q?i֯}?????Unknown
?_HostBiasAddGrad"9gradient_tape/sequential_38/dense_130/BiasAdd/BiasAddGrad(1B`??"[@9B`??"[@AB`??"[@IB`??"[@a?x???jQ?i???|b????Unknown
?`HostCast"3binary_crossentropy/weighted_loss/num_elements/Cast(1??ʡE@9??ʡE@A??ʡE@I??ʡE@a???.UQ?i?l.???Unknown
?aHostGreaterEqual".binary_crossentropy/logistic_loss/GreaterEqual(1d;?O??@9d;?O??@Ad;?O??@Id;?O??@a?q?L}?P?i???R`???Unknown
?bHostBiasAddGrad"9gradient_tape/sequential_38/dense_129/BiasAdd/BiasAddGrad(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a-???^P?i!T,?????Unknown
vcHostAssignAddVariableOp"AssignAddVariableOp_4(1+??N@9+??N@A+??N@I+??N@a?&???]P?i???]? ???Unknown
vdHostMul"%binary_crossentropy/logistic_loss/mul(1??ʡE@9??ʡE@A??ʡE@I??ʡE@aӞ?HTP?i????(???Unknown
?eHostReadVariableOp"-sequential_38/dense_128/MatMul/ReadVariableOp(1#??~j?@9#??~j?@A#??~j?@I#??~j?@agɥ???O?i??.??0???Unknown
XfHostCast"Cast_3(1/?$??@9/?$??@A/?$??@I/?$??@aFų=??O?i?4>=?8???Unknown
?gHostReadVariableOp"-sequential_38/dense_130/MatMul/ReadVariableOp(1??C?l@9??C?l@A??C?l@I??C?l@a???ՈO?i]??r?@???Unknown
vhHostReadVariableOp"Adam/Cast_3/ReadVariableOp(1??K7?A@9??K7?A@A??K7?A@I??K7?A@a6?֬]O?i`??݄H???Unknown
eiHostAddN"Adam/gradients/AddN(1??K7?A@9??K7?A@A??K7?A@I??K7?A@a6?֬]O?icDI\P???Unknown
?jHostReluGrad".gradient_tape/sequential_38/dense_130/ReluGrad(1V-?@9V-?@AV-?@IV-?@a?v???L?i?3?I?W???Unknown
bkHostDivNoNan"div_no_nan_1(1??|?5^@9??|?5^@A??|?5^@I??|?5^@aξ?D?wL?i1]?9?^???Unknown
olHostReadVariableOp"Adam/ReadVariableOp(1? ?rh?@9? ?rh?@A? ?rh?@I? ?rh?@a?/@:?K?i=-ȗe???Unknown
?mHostReadVariableOp"-sequential_38/dense_129/MatMul/ReadVariableOp(15^?I@95^?I@A5^?I@I5^?I@a?~??\K?i?I_^l???Unknown
?nHostNeg"7gradient_tape/binary_crossentropy/logistic_loss/sub/Neg(1?I+?
@9?I+?
@A?I+?
@I?I+?
@a?????J?i??"s???Unknown
XoHostCast"Cast_5(1㥛? ?@9㥛? ?@A㥛? ?@I㥛? ?@a???Z?H?i ?0?7y???Unknown
XpHostEqual"Equal(1?I+?@9?I+?@A?I+?@I?I+?@aI?W@?H?i??1	_???Unknown
?qHostMul"7gradient_tape/binary_crossentropy/logistic_loss/mul/Mul(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@aRE???H?i8CV??????Unknown
?rHostSum"3gradient_tape/binary_crossentropy/logistic_loss/Sum(1ffffff@9ffffff@Affffff@Iffffff@a>??ow{G?i?92?_????Unknown
?sHostBiasAddGrad"9gradient_tape/sequential_38/dense_128/BiasAdd/BiasAddGrad(1333333@9333333@A333333@I333333@a??«/GF?in*\?????Unknown
?tHostBiasAddGrad"9gradient_tape/sequential_38/dense_131/BiasAdd/BiasAddGrad(1?z?G?@9?z?G?@A?z?G?@I?z?G?@a?lff??E?i	Ķ?n????Unknown
?uHostReluGrad".gradient_tape/sequential_38/dense_128/ReluGrad(1???Q?@9???Q?@A???Q?@I???Q?@a??????E?i*????????Unknown
?vHostReluGrad".gradient_tape/sequential_38/dense_133/ReluGrad(1{?G?z@9{?G?z@A{?G?z@I{?G?z@a|???7?E?i?? E????Unknown
?wHostReadVariableOp".sequential_38/dense_129/BiasAdd/ReadVariableOp(1ffffff@9ffffff@Affffff@Iffffff@aɰ\~?yE?i?6;??????Unknown
vxHostExp"%binary_crossentropy/logistic_loss/Exp(1??Q??@9??Q??@A??Q??@I??Q??@a-????D?iJ;qɫ???Unknown
|yHostSelect"(binary_crossentropy/logistic_loss/Select(1?(\???@9?(\???@A?(\???@I?(\???@a8|?ƺD?i)??_˰???Unknown
VzHostCast"Cast(1?n???@9?n???@A?n???@I?n???@a?????C?i?#m?µ???Unknown
?{HostReadVariableOp"-sequential_38/dense_134/MatMul/ReadVariableOp(1=
ףp=@9=
ףp=@A=
ףp=@I=
ףp=@a??0??NC?i̯'5?????Unknown
?|HostAddV2"3gradient_tape/binary_crossentropy/logistic_loss/add(1?G?z@9?G?z@A?G?z@I?G?z@a???G?%C?i??9?_????Unknown
?}HostFloorDiv"*gradient_tape/binary_crossentropy/floordiv(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@aq~S ?B?ipN?????Unknown
?~HostReadVariableOp"-sequential_38/dense_131/MatMul/ReadVariableOp(1ףp=
?@9ףp=
?@Aףp=
?@Iףp=
?@aq~S ?B?i?Oc??????Unknown
vHostReadVariableOp"Adam/Cast_2/ReadVariableOp(1?x?&1@9?x?&1@A?x?&1@I?x?&1@a0?\mB?i?f??Y????Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_1(1d;?O?? @9d;?O?? @Ad;?O?? @Id;?O?? @a?q?L}?@?i?v?Y?????Unknown
?HostRealDiv")gradient_tape/binary_crossentropy/truediv(1?$??C??9?$??C??A?$??C??I?$??C??a{?W??_??i??VQo????Unknown
~?HostDivNoNan"'binary_crossentropy/weighted_loss/value(1m???????9m???????Am???????Im???????a??:@??=?i???
-????Unknown
w?HostAssignAddVariableOp"AssignAddVariableOp_3(1?I+???9?I+???A?I+???I?I+???a3?S?٠<?iis?%?????Unknown
U?HostMul"Mul(1??C?l???9??C?l???A??C?l???I??C?l???a(??? <?i?uo7A????Unknown
y?HostCast"&gradient_tape/binary_crossentropy/Cast(1? ?rh???9? ?rh???A? ?rh???I? ?rh???a?/@:?;?i?]?~?????Unknown
??HostSum"5gradient_tape/binary_crossentropy/logistic_loss/Sum_1(1㥛? ???9㥛? ???A㥛? ???I㥛? ???a???Z?8?i?>J?????Unknown
??Host
Reciprocal":gradient_tape/binary_crossentropy/logistic_loss/Reciprocal(1?O??n??9?O??n??A?O??n??I?O??n??a.??4'7?i?V?0?????Unknown
b?HostIdentity"Identity(1!?rh????9!?rh????A!?rh????I!?rh????aS???57?i??=w?????Unknown?
??HostReluGrad".gradient_tape/sequential_38/dense_131/ReluGrad(1?ʡE????9?ʡE????A?ʡE????I?ʡE????ah|?y6?i?HpfU????Unknown
??Host	ZerosLike"<gradient_tape/binary_crossentropy/logistic_loss/zeros_like_1(1;?O??n??9;?O??n??A;?O??n??I;?O??n??a?m?/2?i??cL?????Unknown
??HostNeg"3gradient_tape/binary_crossentropy/logistic_loss/Neg(1bX9????9bX9????AbX9????IbX9????aK6Nk?.?i????????Unknown
??HostSum"9gradient_tape/binary_crossentropy/logistic_loss/sub/Sum_1(1m???????9m???????Am???????Im???????a??:@??-?iC??or????Unknown
x?HostReadVariableOp"div_no_nan_1/ReadVariableOp(1w??/???9w??/???Aw??/???Iw??/???a?E'p+?,?i???A????Unknown
z?HostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a8$S?+?i?G?g????Unknown
??HostReluGrad".gradient_tape/sequential_38/dense_132/ReluGrad(1D?l?????9D?l?????AD?l?????ID?l?????a?	???)?i?y?????Unknown
??HostSum"7gradient_tape/binary_crossentropy/logistic_loss/mul/Sum(1?Zd;???9?Zd;???A?Zd;???I?Zd;???a@?,S?#?i?#??????Unknown
??HostMul"3gradient_tape/binary_crossentropy/logistic_loss/mul(1????x???9????x???A????x???I????x???a?)??"?iiU
3????Unknown
Y?HostCast"Cast_4(1??ʡE??9??ʡE??A??ʡE??I??ʡE??aH?a?V"?i??Y?1????Unknown
x?HostReadVariableOp"div_no_nan/ReadVariableOp_1(1????????9????????A????????I????????a`Hb???i??,?????Unknown
??HostMul"5gradient_tape/binary_crossentropy/logistic_loss/mul_1(1????????9????????A????????I????????a`Hb???i     ???Unknown2CPU