��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
Adam/v/dense_23/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_23/bias/*
dtype0*
shape:$*%
shared_nameAdam/v/dense_23/bias
y
(Adam/v/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/bias*
_output_shapes
:$*
dtype0
�
Adam/m/dense_23/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_23/bias/*
dtype0*
shape:$*%
shared_nameAdam/m/dense_23/bias
y
(Adam/m/dense_23/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/bias*
_output_shapes
:$*
dtype0
�
Adam/v/dense_23/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_23/kernel/*
dtype0*
shape:	�$*'
shared_nameAdam/v/dense_23/kernel
�
*Adam/v/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_23/kernel*
_output_shapes
:	�$*
dtype0
�
Adam/m/dense_23/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_23/kernel/*
dtype0*
shape:	�$*'
shared_nameAdam/m/dense_23/kernel
�
*Adam/m/dense_23/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_23/kernel*
_output_shapes
:	�$*
dtype0
�
Adam/v/dense_22/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/v/dense_22/bias/*
dtype0*
shape:�*%
shared_nameAdam/v/dense_22/bias
z
(Adam/v/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/m/dense_22/biasVarHandleOp*
_output_shapes
: *%

debug_nameAdam/m/dense_22/bias/*
dtype0*
shape:�*%
shared_nameAdam/m/dense_22/bias
z
(Adam/m/dense_22/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/bias*
_output_shapes	
:�*
dtype0
�
Adam/v/dense_22/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/v/dense_22/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/v/dense_22/kernel
�
*Adam/v/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
Adam/m/dense_22/kernelVarHandleOp*
_output_shapes
: *'

debug_nameAdam/m/dense_22/kernel/*
dtype0*
shape:
��*'
shared_nameAdam/m/dense_22/kernel
�
*Adam/m/dense_22/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense_22/kernel* 
_output_shapes
:
��*
dtype0
�
"Adam/v/batch_normalization_35/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/batch_normalization_35/beta/*
dtype0*
shape:*3
shared_name$"Adam/v/batch_normalization_35/beta
�
6Adam/v/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_35/beta*
_output_shapes
:*
dtype0
�
"Adam/m/batch_normalization_35/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/batch_normalization_35/beta/*
dtype0*
shape:*3
shared_name$"Adam/m/batch_normalization_35/beta
�
6Adam/m/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_35/beta*
_output_shapes
:*
dtype0
�
#Adam/v/batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_35/gamma/*
dtype0*
shape:*4
shared_name%#Adam/v/batch_normalization_35/gamma
�
7Adam/v/batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_35/gamma*
_output_shapes
:*
dtype0
�
#Adam/m/batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_35/gamma/*
dtype0*
shape:*4
shared_name%#Adam/m/batch_normalization_35/gamma
�
7Adam/m/batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_35/gamma*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_35/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_35/bias/*
dtype0*
shape:*&
shared_nameAdam/v/conv2d_35/bias
{
)Adam/v/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/bias*
_output_shapes
:*
dtype0
�
Adam/m/conv2d_35/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_35/bias/*
dtype0*
shape:*&
shared_nameAdam/m/conv2d_35/bias
{
)Adam/m/conv2d_35/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/bias*
_output_shapes
:*
dtype0
�
Adam/v/conv2d_35/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_35/kernel/*
dtype0*
shape:2*(
shared_nameAdam/v/conv2d_35/kernel
�
+Adam/v/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_35/kernel*&
_output_shapes
:2*
dtype0
�
Adam/m/conv2d_35/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_35/kernel/*
dtype0*
shape:2*(
shared_nameAdam/m/conv2d_35/kernel
�
+Adam/m/conv2d_35/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_35/kernel*&
_output_shapes
:2*
dtype0
�
"Adam/v/batch_normalization_34/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/batch_normalization_34/beta/*
dtype0*
shape:2*3
shared_name$"Adam/v/batch_normalization_34/beta
�
6Adam/v/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_34/beta*
_output_shapes
:2*
dtype0
�
"Adam/m/batch_normalization_34/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/batch_normalization_34/beta/*
dtype0*
shape:2*3
shared_name$"Adam/m/batch_normalization_34/beta
�
6Adam/m/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_34/beta*
_output_shapes
:2*
dtype0
�
#Adam/v/batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_34/gamma/*
dtype0*
shape:2*4
shared_name%#Adam/v/batch_normalization_34/gamma
�
7Adam/v/batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_34/gamma*
_output_shapes
:2*
dtype0
�
#Adam/m/batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_34/gamma/*
dtype0*
shape:2*4
shared_name%#Adam/m/batch_normalization_34/gamma
�
7Adam/m/batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_34/gamma*
_output_shapes
:2*
dtype0
�
Adam/v/conv2d_34/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_34/bias/*
dtype0*
shape:2*&
shared_nameAdam/v/conv2d_34/bias
{
)Adam/v/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/bias*
_output_shapes
:2*
dtype0
�
Adam/m/conv2d_34/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_34/bias/*
dtype0*
shape:2*&
shared_nameAdam/m/conv2d_34/bias
{
)Adam/m/conv2d_34/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/bias*
_output_shapes
:2*
dtype0
�
Adam/v/conv2d_34/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_34/kernel/*
dtype0*
shape:K2*(
shared_nameAdam/v/conv2d_34/kernel
�
+Adam/v/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_34/kernel*&
_output_shapes
:K2*
dtype0
�
Adam/m/conv2d_34/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_34/kernel/*
dtype0*
shape:K2*(
shared_nameAdam/m/conv2d_34/kernel
�
+Adam/m/conv2d_34/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_34/kernel*&
_output_shapes
:K2*
dtype0
�
"Adam/v/batch_normalization_33/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/v/batch_normalization_33/beta/*
dtype0*
shape:K*3
shared_name$"Adam/v/batch_normalization_33/beta
�
6Adam/v/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOp"Adam/v/batch_normalization_33/beta*
_output_shapes
:K*
dtype0
�
"Adam/m/batch_normalization_33/betaVarHandleOp*
_output_shapes
: *3

debug_name%#Adam/m/batch_normalization_33/beta/*
dtype0*
shape:K*3
shared_name$"Adam/m/batch_normalization_33/beta
�
6Adam/m/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOp"Adam/m/batch_normalization_33/beta*
_output_shapes
:K*
dtype0
�
#Adam/v/batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/v/batch_normalization_33/gamma/*
dtype0*
shape:K*4
shared_name%#Adam/v/batch_normalization_33/gamma
�
7Adam/v/batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOp#Adam/v/batch_normalization_33/gamma*
_output_shapes
:K*
dtype0
�
#Adam/m/batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *4

debug_name&$Adam/m/batch_normalization_33/gamma/*
dtype0*
shape:K*4
shared_name%#Adam/m/batch_normalization_33/gamma
�
7Adam/m/batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOp#Adam/m/batch_normalization_33/gamma*
_output_shapes
:K*
dtype0
�
Adam/v/conv2d_33/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/v/conv2d_33/bias/*
dtype0*
shape:K*&
shared_nameAdam/v/conv2d_33/bias
{
)Adam/v/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/bias*
_output_shapes
:K*
dtype0
�
Adam/m/conv2d_33/biasVarHandleOp*
_output_shapes
: *&

debug_nameAdam/m/conv2d_33/bias/*
dtype0*
shape:K*&
shared_nameAdam/m/conv2d_33/bias
{
)Adam/m/conv2d_33/bias/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/bias*
_output_shapes
:K*
dtype0
�
Adam/v/conv2d_33/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/v/conv2d_33/kernel/*
dtype0*
shape:K*(
shared_nameAdam/v/conv2d_33/kernel
�
+Adam/v/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/v/conv2d_33/kernel*&
_output_shapes
:K*
dtype0
�
Adam/m/conv2d_33/kernelVarHandleOp*
_output_shapes
: *(

debug_nameAdam/m/conv2d_33/kernel/*
dtype0*
shape:K*(
shared_nameAdam/m/conv2d_33/kernel
�
+Adam/m/conv2d_33/kernel/Read/ReadVariableOpReadVariableOpAdam/m/conv2d_33/kernel*&
_output_shapes
:K*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
dense_23/biasVarHandleOp*
_output_shapes
: *

debug_namedense_23/bias/*
dtype0*
shape:$*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:$*
dtype0
�
dense_23/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_23/kernel/*
dtype0*
shape:	�$* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	�$*
dtype0
�
dense_22/biasVarHandleOp*
_output_shapes
: *

debug_namedense_22/bias/*
dtype0*
shape:�*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:�*
dtype0
�
dense_22/kernelVarHandleOp*
_output_shapes
: * 

debug_namedense_22/kernel/*
dtype0*
shape:
��* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
��*
dtype0
�
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_35/moving_variance/*
dtype0*
shape:*7
shared_name(&batch_normalization_35/moving_variance
�
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes
:*
dtype0
�
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_35/moving_mean/*
dtype0*
shape:*3
shared_name$"batch_normalization_35/moving_mean
�
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes
:*
dtype0
�
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_35/beta/*
dtype0*
shape:*,
shared_namebatch_normalization_35/beta
�
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes
:*
dtype0
�
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_35/gamma/*
dtype0*
shape:*-
shared_namebatch_normalization_35/gamma
�
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes
:*
dtype0
�
conv2d_35/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_35/bias/*
dtype0*
shape:*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:*
dtype0
�
conv2d_35/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_35/kernel/*
dtype0*
shape:2*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:2*
dtype0
�
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_34/moving_variance/*
dtype0*
shape:2*7
shared_name(&batch_normalization_34/moving_variance
�
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes
:2*
dtype0
�
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_34/moving_mean/*
dtype0*
shape:2*3
shared_name$"batch_normalization_34/moving_mean
�
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes
:2*
dtype0
�
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_34/beta/*
dtype0*
shape:2*,
shared_namebatch_normalization_34/beta
�
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes
:2*
dtype0
�
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_34/gamma/*
dtype0*
shape:2*-
shared_namebatch_normalization_34/gamma
�
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes
:2*
dtype0
�
conv2d_34/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_34/bias/*
dtype0*
shape:2*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:2*
dtype0
�
conv2d_34/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_34/kernel/*
dtype0*
shape:K2*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:K2*
dtype0
�
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *7

debug_name)'batch_normalization_33/moving_variance/*
dtype0*
shape:K*7
shared_name(&batch_normalization_33/moving_variance
�
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:K*
dtype0
�
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *3

debug_name%#batch_normalization_33/moving_mean/*
dtype0*
shape:K*3
shared_name$"batch_normalization_33/moving_mean
�
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:K*
dtype0
�
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *,

debug_namebatch_normalization_33/beta/*
dtype0*
shape:K*,
shared_namebatch_normalization_33/beta
�
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:K*
dtype0
�
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *-

debug_namebatch_normalization_33/gamma/*
dtype0*
shape:K*-
shared_namebatch_normalization_33/gamma
�
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:K*
dtype0
�
conv2d_33/biasVarHandleOp*
_output_shapes
: *

debug_nameconv2d_33/bias/*
dtype0*
shape:K*
shared_nameconv2d_33/bias
m
"conv2d_33/bias/Read/ReadVariableOpReadVariableOpconv2d_33/bias*
_output_shapes
:K*
dtype0
�
conv2d_33/kernelVarHandleOp*
_output_shapes
: *!

debug_nameconv2d_33/kernel/*
dtype0*
shape:K*!
shared_nameconv2d_33/kernel
}
$conv2d_33/kernel/Read/ReadVariableOpReadVariableOpconv2d_33/kernel*&
_output_shapes
:K*
dtype0
�
serving_default_conv2d_33_inputPlaceholder*/
_output_shapes
:���������@@*
dtype0*$
shape:���������@@
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_33_inputconv2d_33/kernelconv2d_33/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_34/kernelconv2d_34/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference_signature_wrapper_158901

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op*
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'axis
	(gamma
)beta
*moving_mean
+moving_variance*
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op*
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator* 
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance*
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op*
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses* 
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias*
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
0
1
(2
)3
*4
+5
86
97
I8
J9
K10
L11
Y12
Z13
c14
d15
e16
f17
y18
z19
�20
�21*
|
0
1
(2
)3
84
95
I6
J7
Y8
Z9
c10
d11
y12
z13
�14
�15*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla*

�serving_default* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_33/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_33/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
(0
)1
*2
+3*

(0
)1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

80
91*

80
91*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
 
I0
J1
K2
L3*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
`Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
c0
d1
e2
f3*

c0
d1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

y0
z1*

y0
z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
*0
+1
K2
L3
e4
f5*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

�0
�1*
* 
* 
* 
* 
* 
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15*
* 
* 
* 
* 
* 
* 
* 
* 
* 

*0
+1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

K0
L1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

e0
f1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
b\
VARIABLE_VALUEAdam/m/conv2d_33/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/conv2d_33/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/conv2d_33/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/conv2d_33/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/m/batch_normalization_33/gamma1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE#Adam/v/batch_normalization_33/gamma1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/m/batch_normalization_33/beta1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"Adam/v/batch_normalization_33/beta1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/conv2d_34/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_34/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_34/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_34/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_34/gamma2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_34/gamma2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_34/beta2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_34/beta2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/m/conv2d_35/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEAdam/v/conv2d_35/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/m/conv2d_35/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEAdam/v/conv2d_35/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/m/batch_normalization_35/gamma2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE#Adam/v/batch_normalization_35/gamma2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/m/batch_normalization_35/beta2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE"Adam/v/batch_normalization_35/beta2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_22/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_22/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_22/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_22/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/dense_23/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/dense_23/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/m/dense_23/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEAdam/v/dense_23/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameconv2d_33/kernelconv2d_33/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_34/kernelconv2d_34/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	iterationlearning_rateAdam/m/conv2d_33/kernelAdam/v/conv2d_33/kernelAdam/m/conv2d_33/biasAdam/v/conv2d_33/bias#Adam/m/batch_normalization_33/gamma#Adam/v/batch_normalization_33/gamma"Adam/m/batch_normalization_33/beta"Adam/v/batch_normalization_33/betaAdam/m/conv2d_34/kernelAdam/v/conv2d_34/kernelAdam/m/conv2d_34/biasAdam/v/conv2d_34/bias#Adam/m/batch_normalization_34/gamma#Adam/v/batch_normalization_34/gamma"Adam/m/batch_normalization_34/beta"Adam/v/batch_normalization_34/betaAdam/m/conv2d_35/kernelAdam/v/conv2d_35/kernelAdam/m/conv2d_35/biasAdam/v/conv2d_35/bias#Adam/m/batch_normalization_35/gamma#Adam/v/batch_normalization_35/gamma"Adam/m/batch_normalization_35/beta"Adam/v/batch_normalization_35/betaAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biastotal_1count_1totalcountConst*I
TinB
@2>*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__traced_save_159664
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_33/kernelconv2d_33/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_34/kernelconv2d_34/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_35/kernelconv2d_35/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	iterationlearning_rateAdam/m/conv2d_33/kernelAdam/v/conv2d_33/kernelAdam/m/conv2d_33/biasAdam/v/conv2d_33/bias#Adam/m/batch_normalization_33/gamma#Adam/v/batch_normalization_33/gamma"Adam/m/batch_normalization_33/beta"Adam/v/batch_normalization_33/betaAdam/m/conv2d_34/kernelAdam/v/conv2d_34/kernelAdam/m/conv2d_34/biasAdam/v/conv2d_34/bias#Adam/m/batch_normalization_34/gamma#Adam/v/batch_normalization_34/gamma"Adam/m/batch_normalization_34/beta"Adam/v/batch_normalization_34/betaAdam/m/conv2d_35/kernelAdam/v/conv2d_35/kernelAdam/m/conv2d_35/biasAdam/v/conv2d_35/bias#Adam/m/batch_normalization_35/gamma#Adam/v/batch_normalization_35/gamma"Adam/m/batch_normalization_35/beta"Adam/v/batch_normalization_35/betaAdam/m/dense_22/kernelAdam/v/dense_22/kernelAdam/m/dense_22/biasAdam/v/dense_22/biasAdam/m/dense_23/kernelAdam/v/dense_23/kernelAdam/m/dense_23/biasAdam/v/dense_23/biastotal_1count_1totalcount*H
TinA
?2=*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference__traced_restore_159853��
�
�
.__inference_sequential_11_layer_call_fn_158730
conv2d_33_input!
unknown:K
	unknown_0:K
	unknown_1:K
	unknown_2:K
	unknown_3:K
	unknown_4:K#
	unknown_5:K2
	unknown_6:2
	unknown_7:2
	unknown_8:2
	unknown_9:2

unknown_10:2$

unknown_11:2

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:
��

unknown_18:	�

unknown_19:	�$

unknown_20:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_158609o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158726:&"
 
_user_specified_name158724:&"
 
_user_specified_name158722:&"
 
_user_specified_name158720:&"
 
_user_specified_name158718:&"
 
_user_specified_name158716:&"
 
_user_specified_name158714:&"
 
_user_specified_name158712:&"
 
_user_specified_name158710:&"
 
_user_specified_name158708:&"
 
_user_specified_name158706:&"
 
_user_specified_name158704:&
"
 
_user_specified_name158702:&	"
 
_user_specified_name158700:&"
 
_user_specified_name158698:&"
 
_user_specified_name158696:&"
 
_user_specified_name158694:&"
 
_user_specified_name158692:&"
 
_user_specified_name158690:&"
 
_user_specified_name158688:&"
 
_user_specified_name158686:&"
 
_user_specified_name158684:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�
G
+__inference_dropout_22_layer_call_fn_159023

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_158636h
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������  2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  2:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158313

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_159235

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_159040

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  2c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������  2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  2:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�
�
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158475

inputs8
conv2d_readvariableop_resource:K-
biasadd_readvariableop_resource:K
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:K*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@K*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:K*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@KX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@Ki
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@KS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158264

inputs%
readvariableop_resource:K'
readvariableop_1_resource:K6
(fusedbatchnormv3_readvariableop_resource:K8
*fusedbatchnormv3_readvariableop_1_resource:K
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:K*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:K*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������K:K:K:K:K:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�
�
)__inference_dense_23_layer_call_fn_159271

inputs
unknown:	�$
	unknown_0:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_158602o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159267:&"
 
_user_specified_name159265:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_34_layer_call_fn_159066

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158354�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159062:&"
 
_user_specified_name159060:&"
 
_user_specified_name159058:&"
 
_user_specified_name159056:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
�
E__inference_conv2d_35_layer_call_and_return_conditional_losses_158540

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
E__inference_conv2d_34_layer_call_and_return_conditional_losses_158501

inputs8
conv2d_readvariableop_resource:K2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:K2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  K: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������  K
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159084

inputs%
readvariableop_resource:2'
readvariableop_1_resource:26
(fusedbatchnormv3_readvariableop_resource:28
*fusedbatchnormv3_readvariableop_1_resource:2
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_159282

inputs1
matmul_readvariableop_resource:	�$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������$S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_35_layer_call_and_return_conditional_losses_159132

inputs8
conv2d_readvariableop_resource:2-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158336

inputs%
readvariableop_resource:2'
readvariableop_1_resource:26
(fusedbatchnormv3_readvariableop_resource:28
*fusedbatchnormv3_readvariableop_1_resource:2
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
G
+__inference_dropout_23_layer_call_fn_159245

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_158673a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_35_layer_call_fn_159145

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158408�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159141:&"
 
_user_specified_name159139:&"
 
_user_specified_name159137:&"
 
_user_specified_name159135:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_34_layer_call_and_return_conditional_losses_159013

inputs8
conv2d_readvariableop_resource:K2-
biasadd_readvariableop_resource:2
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:K2*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������  2i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������  2S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  K: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������  K
 
_user_specified_nameinputs
�
�
*__inference_conv2d_33_layer_call_fn_158910

inputs!
unknown:K
	unknown_0:K
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158475w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������@@K<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158906:&"
 
_user_specified_name158904:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159102

inputs%
readvariableop_resource:2'
readvariableop_1_resource:26
(fusedbatchnormv3_readvariableop_resource:28
*fusedbatchnormv3_readvariableop_1_resource:2
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_159215

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159194

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_33_layer_call_fn_158988

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158313�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158408

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_33_layer_call_fn_158947

inputs
unknown:K
	unknown_0:K
	unknown_1:K
	unknown_2:K
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������K*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158282�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158943:&"
 
_user_specified_name158941:&"
 
_user_specified_name158939:&"
 
_user_specified_name158937:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_159204

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_158385

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_158457

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
*__inference_conv2d_35_layer_call_fn_159121

inputs!
unknown:2
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_158540w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������2: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159117:&"
 
_user_specified_name159115:W S
/
_output_shapes
:���������2
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_35_layer_call_fn_159199

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_158457�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158354

inputs%
readvariableop_resource:2'
readvariableop_1_resource:26
(fusedbatchnormv3_readvariableop_resource:28
*fusedbatchnormv3_readvariableop_1_resource:2
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:2*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������2:2:2:2:2:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
�
d
+__inference_dropout_22_layer_call_fn_159018

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_158518w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  222
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�
�
)__inference_dense_22_layer_call_fn_159224

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_158573p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159220:&"
 
_user_specified_name159218:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158921

inputs8
conv2d_readvariableop_resource:K-
biasadd_readvariableop_resource:K
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:K*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@K*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:K*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@KX
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������@@Ki
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������@@KS
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������@@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������@@
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_33_layer_call_fn_158934

inputs
unknown:K
	unknown_0:K
	unknown_1:K
	unknown_2:K
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158264�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158930:&"
 
_user_specified_name158928:&"
 
_user_specified_name158926:&"
 
_user_specified_name158924:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�
�
!__inference__wrapped_model_158246
conv2d_33_inputP
6sequential_11_conv2d_33_conv2d_readvariableop_resource:KE
7sequential_11_conv2d_33_biasadd_readvariableop_resource:KJ
<sequential_11_batch_normalization_33_readvariableop_resource:KL
>sequential_11_batch_normalization_33_readvariableop_1_resource:K[
Msequential_11_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:K]
Osequential_11_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:KP
6sequential_11_conv2d_34_conv2d_readvariableop_resource:K2E
7sequential_11_conv2d_34_biasadd_readvariableop_resource:2J
<sequential_11_batch_normalization_34_readvariableop_resource:2L
>sequential_11_batch_normalization_34_readvariableop_1_resource:2[
Msequential_11_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:2]
Osequential_11_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:2P
6sequential_11_conv2d_35_conv2d_readvariableop_resource:2E
7sequential_11_conv2d_35_biasadd_readvariableop_resource:J
<sequential_11_batch_normalization_35_readvariableop_resource:L
>sequential_11_batch_normalization_35_readvariableop_1_resource:[
Msequential_11_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:]
Osequential_11_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:I
5sequential_11_dense_22_matmul_readvariableop_resource:
��E
6sequential_11_dense_22_biasadd_readvariableop_resource:	�H
5sequential_11_dense_23_matmul_readvariableop_resource:	�$D
6sequential_11_dense_23_biasadd_readvariableop_resource:$
identity��Dsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_33/ReadVariableOp�5sequential_11/batch_normalization_33/ReadVariableOp_1�Dsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_34/ReadVariableOp�5sequential_11/batch_normalization_34/ReadVariableOp_1�Dsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp�Fsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1�3sequential_11/batch_normalization_35/ReadVariableOp�5sequential_11/batch_normalization_35/ReadVariableOp_1�.sequential_11/conv2d_33/BiasAdd/ReadVariableOp�-sequential_11/conv2d_33/Conv2D/ReadVariableOp�.sequential_11/conv2d_34/BiasAdd/ReadVariableOp�-sequential_11/conv2d_34/Conv2D/ReadVariableOp�.sequential_11/conv2d_35/BiasAdd/ReadVariableOp�-sequential_11/conv2d_35/Conv2D/ReadVariableOp�-sequential_11/dense_22/BiasAdd/ReadVariableOp�,sequential_11/dense_22/MatMul/ReadVariableOp�-sequential_11/dense_23/BiasAdd/ReadVariableOp�,sequential_11/dense_23/MatMul/ReadVariableOp�
-sequential_11/conv2d_33/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_33_conv2d_readvariableop_resource*&
_output_shapes
:K*
dtype0�
sequential_11/conv2d_33/Conv2DConv2Dconv2d_33_input5sequential_11/conv2d_33/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@K*
paddingSAME*
strides
�
.sequential_11/conv2d_33/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_33_biasadd_readvariableop_resource*
_output_shapes
:K*
dtype0�
sequential_11/conv2d_33/BiasAddBiasAdd'sequential_11/conv2d_33/Conv2D:output:06sequential_11/conv2d_33/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@@K�
sequential_11/conv2d_33/ReluRelu(sequential_11/conv2d_33/BiasAdd:output:0*
T0*/
_output_shapes
:���������@@K�
3sequential_11/batch_normalization_33/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_33_readvariableop_resource*
_output_shapes
:K*
dtype0�
5sequential_11/batch_normalization_33/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
Dsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:K*
dtype0�
Fsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
5sequential_11/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_33/Relu:activations:0;sequential_11/batch_normalization_33/ReadVariableOp:value:0=sequential_11/batch_normalization_33/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@@K:K:K:K:K:*
epsilon%o�:*
is_training( �
&sequential_11/max_pooling2d_33/MaxPoolMaxPool9sequential_11/batch_normalization_33/FusedBatchNormV3:y:0*/
_output_shapes
:���������  K*
ksize
*
paddingSAME*
strides
�
-sequential_11/conv2d_34/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:K2*
dtype0�
sequential_11/conv2d_34/Conv2DConv2D/sequential_11/max_pooling2d_33/MaxPool:output:05sequential_11/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2*
paddingSAME*
strides
�
.sequential_11/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0�
sequential_11/conv2d_34/BiasAddBiasAdd'sequential_11/conv2d_34/Conv2D:output:06sequential_11/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������  2�
sequential_11/conv2d_34/ReluRelu(sequential_11/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:���������  2�
!sequential_11/dropout_22/IdentityIdentity*sequential_11/conv2d_34/Relu:activations:0*
T0*/
_output_shapes
:���������  2�
3sequential_11/batch_normalization_34/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_34_readvariableop_resource*
_output_shapes
:2*
dtype0�
5sequential_11/batch_normalization_34/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_34_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
Dsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:2*
dtype0�
Fsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:2*
dtype0�
5sequential_11/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3*sequential_11/dropout_22/Identity:output:0;sequential_11/batch_normalization_34/ReadVariableOp:value:0=sequential_11/batch_normalization_34/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������  2:2:2:2:2:*
epsilon%o�:*
is_training( �
&sequential_11/max_pooling2d_34/MaxPoolMaxPool9sequential_11/batch_normalization_34/FusedBatchNormV3:y:0*/
_output_shapes
:���������2*
ksize
*
paddingSAME*
strides
�
-sequential_11/conv2d_35/Conv2D/ReadVariableOpReadVariableOp6sequential_11_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:2*
dtype0�
sequential_11/conv2d_35/Conv2DConv2D/sequential_11/max_pooling2d_34/MaxPool:output:05sequential_11/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
�
.sequential_11/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp7sequential_11_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
sequential_11/conv2d_35/BiasAddBiasAdd'sequential_11/conv2d_35/Conv2D:output:06sequential_11/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:����������
sequential_11/conv2d_35/ReluRelu(sequential_11/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:����������
3sequential_11/batch_normalization_35/ReadVariableOpReadVariableOp<sequential_11_batch_normalization_35_readvariableop_resource*
_output_shapes
:*
dtype0�
5sequential_11/batch_normalization_35/ReadVariableOp_1ReadVariableOp>sequential_11_batch_normalization_35_readvariableop_1_resource*
_output_shapes
:*
dtype0�
Dsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_11_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
Fsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_11_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
5sequential_11/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3*sequential_11/conv2d_35/Relu:activations:0;sequential_11/batch_normalization_35/ReadVariableOp:value:0=sequential_11/batch_normalization_35/ReadVariableOp_1:value:0Lsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������:::::*
epsilon%o�:*
is_training( �
&sequential_11/max_pooling2d_35/MaxPoolMaxPool9sequential_11/batch_normalization_35/FusedBatchNormV3:y:0*/
_output_shapes
:���������*
ksize
*
paddingSAME*
strides
o
sequential_11/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  �
 sequential_11/flatten_11/ReshapeReshape/sequential_11/max_pooling2d_35/MaxPool:output:0'sequential_11/flatten_11/Const:output:0*
T0*(
_output_shapes
:�����������
,sequential_11/dense_22/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
sequential_11/dense_22/MatMulMatMul)sequential_11/flatten_11/Reshape:output:04sequential_11/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
-sequential_11/dense_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
sequential_11/dense_22/BiasAddBiasAdd'sequential_11/dense_22/MatMul:product:05sequential_11/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
sequential_11/dense_22/ReluRelu'sequential_11/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
!sequential_11/dropout_23/IdentityIdentity)sequential_11/dense_22/Relu:activations:0*
T0*(
_output_shapes
:�����������
,sequential_11/dense_23/MatMul/ReadVariableOpReadVariableOp5sequential_11_dense_23_matmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0�
sequential_11/dense_23/MatMulMatMul*sequential_11/dropout_23/Identity:output:04sequential_11/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
-sequential_11/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_11_dense_23_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype0�
sequential_11/dense_23/BiasAddBiasAdd'sequential_11/dense_23/MatMul:product:05sequential_11/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$�
sequential_11/dense_23/SoftmaxSoftmax'sequential_11/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:���������$w
IdentityIdentity(sequential_11/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������$�	
NoOpNoOpE^sequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_33/ReadVariableOp6^sequential_11/batch_normalization_33/ReadVariableOp_1E^sequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_34/ReadVariableOp6^sequential_11/batch_normalization_34/ReadVariableOp_1E^sequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOpG^sequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_14^sequential_11/batch_normalization_35/ReadVariableOp6^sequential_11/batch_normalization_35/ReadVariableOp_1/^sequential_11/conv2d_33/BiasAdd/ReadVariableOp.^sequential_11/conv2d_33/Conv2D/ReadVariableOp/^sequential_11/conv2d_34/BiasAdd/ReadVariableOp.^sequential_11/conv2d_34/Conv2D/ReadVariableOp/^sequential_11/conv2d_35/BiasAdd/ReadVariableOp.^sequential_11/conv2d_35/Conv2D/ReadVariableOp.^sequential_11/dense_22/BiasAdd/ReadVariableOp-^sequential_11/dense_22/MatMul/ReadVariableOp.^sequential_11/dense_23/BiasAdd/ReadVariableOp-^sequential_11/dense_23/MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 2�
Fsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12�
Dsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2n
5sequential_11/batch_normalization_33/ReadVariableOp_15sequential_11/batch_normalization_33/ReadVariableOp_12j
3sequential_11/batch_normalization_33/ReadVariableOp3sequential_11/batch_normalization_33/ReadVariableOp2�
Fsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12�
Dsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2n
5sequential_11/batch_normalization_34/ReadVariableOp_15sequential_11/batch_normalization_34/ReadVariableOp_12j
3sequential_11/batch_normalization_34/ReadVariableOp3sequential_11/batch_normalization_34/ReadVariableOp2�
Fsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Fsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12�
Dsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOpDsequential_11/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2n
5sequential_11/batch_normalization_35/ReadVariableOp_15sequential_11/batch_normalization_35/ReadVariableOp_12j
3sequential_11/batch_normalization_35/ReadVariableOp3sequential_11/batch_normalization_35/ReadVariableOp2`
.sequential_11/conv2d_33/BiasAdd/ReadVariableOp.sequential_11/conv2d_33/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_33/Conv2D/ReadVariableOp-sequential_11/conv2d_33/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_34/BiasAdd/ReadVariableOp.sequential_11/conv2d_34/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_34/Conv2D/ReadVariableOp-sequential_11/conv2d_34/Conv2D/ReadVariableOp2`
.sequential_11/conv2d_35/BiasAdd/ReadVariableOp.sequential_11/conv2d_35/BiasAdd/ReadVariableOp2^
-sequential_11/conv2d_35/Conv2D/ReadVariableOp-sequential_11/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_11/dense_22/BiasAdd/ReadVariableOp-sequential_11/dense_22/BiasAdd/ReadVariableOp2\
,sequential_11/dense_22/MatMul/ReadVariableOp,sequential_11/dense_22/MatMul/ReadVariableOp2^
-sequential_11/dense_23/BiasAdd/ReadVariableOp-sequential_11/dense_23/BiasAdd/ReadVariableOp2\
,sequential_11/dense_23/MatMul/ReadVariableOp,sequential_11/dense_23/MatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�
�
*__inference_conv2d_34_layer_call_fn_159002

inputs!
unknown:K2
	unknown_0:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_158501w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������  2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������  K: : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158998:&"
 
_user_specified_name158996:W S
/
_output_shapes
:���������  K
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158965

inputs%
readvariableop_resource:K'
readvariableop_1_resource:K6
(fusedbatchnormv3_readvariableop_resource:K8
*fusedbatchnormv3_readvariableop_1_resource:K
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:K*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:K*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������K:K:K:K:K:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�
h
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158993

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_159262

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_158518

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������  2Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������  2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������  2T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������  2i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������  2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  2:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_159257

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
.__inference_sequential_11_layer_call_fn_158779
conv2d_33_input!
unknown:K
	unknown_0:K
	unknown_1:K
	unknown_2:K
	unknown_3:K
	unknown_4:K#
	unknown_5:K2
	unknown_6:2
	unknown_7:2
	unknown_8:2
	unknown_9:2

unknown_10:2$

unknown_11:2

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:
��

unknown_18:	�

unknown_19:	�$

unknown_20:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *R
fMRK
I__inference_sequential_11_layer_call_and_return_conditional_losses_158681o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158775:&"
 
_user_specified_name158773:&"
 
_user_specified_name158771:&"
 
_user_specified_name158769:&"
 
_user_specified_name158767:&"
 
_user_specified_name158765:&"
 
_user_specified_name158763:&"
 
_user_specified_name158761:&"
 
_user_specified_name158759:&"
 
_user_specified_name158757:&"
 
_user_specified_name158755:&"
 
_user_specified_name158753:&
"
 
_user_specified_name158751:&	"
 
_user_specified_name158749:&"
 
_user_specified_name158747:&"
 
_user_specified_name158745:&"
 
_user_specified_name158743:&"
 
_user_specified_name158741:&"
 
_user_specified_name158739:&"
 
_user_specified_name158737:&"
 
_user_specified_name158735:&"
 
_user_specified_name158733:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�
�
$__inference_signature_wrapper_158901
conv2d_33_input!
unknown:K
	unknown_0:K
	unknown_1:K
	unknown_2:K
	unknown_3:K
	unknown_4:K#
	unknown_5:K2
	unknown_6:2
	unknown_7:2
	unknown_8:2
	unknown_9:2

unknown_10:2$

unknown_11:2

unknown_12:

unknown_13:

unknown_14:

unknown_15:

unknown_16:

unknown_17:
��

unknown_18:	�

unknown_19:	�$

unknown_20:$
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallconv2d_33_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*8
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__wrapped_model_158246o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name158897:&"
 
_user_specified_name158895:&"
 
_user_specified_name158893:&"
 
_user_specified_name158891:&"
 
_user_specified_name158889:&"
 
_user_specified_name158887:&"
 
_user_specified_name158885:&"
 
_user_specified_name158883:&"
 
_user_specified_name158881:&"
 
_user_specified_name158879:&"
 
_user_specified_name158877:&"
 
_user_specified_name158875:&
"
 
_user_specified_name158873:&	"
 
_user_specified_name158871:&"
 
_user_specified_name158869:&"
 
_user_specified_name158867:&"
 
_user_specified_name158865:&"
 
_user_specified_name158863:&"
 
_user_specified_name158861:&"
 
_user_specified_name158859:&"
 
_user_specified_name158857:&"
 
_user_specified_name158855:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�

�
D__inference_dense_22_layer_call_and_return_conditional_losses_158573

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_34_layer_call_fn_159053

inputs
unknown:2
	unknown_0:2
	unknown_1:2
	unknown_2:2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158336�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������2<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������2: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159049:&"
 
_user_specified_name159047:&"
 
_user_specified_name159045:&"
 
_user_specified_name159043:i e
A
_output_shapes/
-:+���������������������������2
 
_user_specified_nameinputs
��
�9
__inference__traced_save_159664
file_prefixA
'read_disablecopyonread_conv2d_33_kernel:K5
'read_1_disablecopyonread_conv2d_33_bias:KC
5read_2_disablecopyonread_batch_normalization_33_gamma:KB
4read_3_disablecopyonread_batch_normalization_33_beta:KI
;read_4_disablecopyonread_batch_normalization_33_moving_mean:KM
?read_5_disablecopyonread_batch_normalization_33_moving_variance:KC
)read_6_disablecopyonread_conv2d_34_kernel:K25
'read_7_disablecopyonread_conv2d_34_bias:2C
5read_8_disablecopyonread_batch_normalization_34_gamma:2B
4read_9_disablecopyonread_batch_normalization_34_beta:2J
<read_10_disablecopyonread_batch_normalization_34_moving_mean:2N
@read_11_disablecopyonread_batch_normalization_34_moving_variance:2D
*read_12_disablecopyonread_conv2d_35_kernel:26
(read_13_disablecopyonread_conv2d_35_bias:D
6read_14_disablecopyonread_batch_normalization_35_gamma:C
5read_15_disablecopyonread_batch_normalization_35_beta:J
<read_16_disablecopyonread_batch_normalization_35_moving_mean:N
@read_17_disablecopyonread_batch_normalization_35_moving_variance:=
)read_18_disablecopyonread_dense_22_kernel:
��6
'read_19_disablecopyonread_dense_22_bias:	�<
)read_20_disablecopyonread_dense_23_kernel:	�$5
'read_21_disablecopyonread_dense_23_bias:$-
#read_22_disablecopyonread_iteration:	 1
'read_23_disablecopyonread_learning_rate: K
1read_24_disablecopyonread_adam_m_conv2d_33_kernel:KK
1read_25_disablecopyonread_adam_v_conv2d_33_kernel:K=
/read_26_disablecopyonread_adam_m_conv2d_33_bias:K=
/read_27_disablecopyonread_adam_v_conv2d_33_bias:KK
=read_28_disablecopyonread_adam_m_batch_normalization_33_gamma:KK
=read_29_disablecopyonread_adam_v_batch_normalization_33_gamma:KJ
<read_30_disablecopyonread_adam_m_batch_normalization_33_beta:KJ
<read_31_disablecopyonread_adam_v_batch_normalization_33_beta:KK
1read_32_disablecopyonread_adam_m_conv2d_34_kernel:K2K
1read_33_disablecopyonread_adam_v_conv2d_34_kernel:K2=
/read_34_disablecopyonread_adam_m_conv2d_34_bias:2=
/read_35_disablecopyonread_adam_v_conv2d_34_bias:2K
=read_36_disablecopyonread_adam_m_batch_normalization_34_gamma:2K
=read_37_disablecopyonread_adam_v_batch_normalization_34_gamma:2J
<read_38_disablecopyonread_adam_m_batch_normalization_34_beta:2J
<read_39_disablecopyonread_adam_v_batch_normalization_34_beta:2K
1read_40_disablecopyonread_adam_m_conv2d_35_kernel:2K
1read_41_disablecopyonread_adam_v_conv2d_35_kernel:2=
/read_42_disablecopyonread_adam_m_conv2d_35_bias:=
/read_43_disablecopyonread_adam_v_conv2d_35_bias:K
=read_44_disablecopyonread_adam_m_batch_normalization_35_gamma:K
=read_45_disablecopyonread_adam_v_batch_normalization_35_gamma:J
<read_46_disablecopyonread_adam_m_batch_normalization_35_beta:J
<read_47_disablecopyonread_adam_v_batch_normalization_35_beta:D
0read_48_disablecopyonread_adam_m_dense_22_kernel:
��D
0read_49_disablecopyonread_adam_v_dense_22_kernel:
��=
.read_50_disablecopyonread_adam_m_dense_22_bias:	�=
.read_51_disablecopyonread_adam_v_dense_22_bias:	�C
0read_52_disablecopyonread_adam_m_dense_23_kernel:	�$C
0read_53_disablecopyonread_adam_v_dense_23_kernel:	�$<
.read_54_disablecopyonread_adam_m_dense_23_bias:$<
.read_55_disablecopyonread_adam_v_dense_23_bias:$+
!read_56_disablecopyonread_total_1: +
!read_57_disablecopyonread_count_1: )
read_58_disablecopyonread_total: )
read_59_disablecopyonread_count: 
savev2_const
identity_121��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: y
Read/DisableCopyOnReadDisableCopyOnRead'read_disablecopyonread_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp'read_disablecopyonread_conv2d_33_kernel^Read/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K*
dtype0q
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Ki

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*&
_output_shapes
:K{
Read_1/DisableCopyOnReadDisableCopyOnRead'read_1_disablecopyonread_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp'read_1_disablecopyonread_conv2d_33_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:K_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_2/DisableCopyOnReadDisableCopyOnRead5read_2_disablecopyonread_batch_normalization_33_gamma"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp5read_2_disablecopyonread_batch_normalization_33_gamma^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0i

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:K_

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_3/DisableCopyOnReadDisableCopyOnRead4read_3_disablecopyonread_batch_normalization_33_beta"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp4read_3_disablecopyonread_batch_normalization_33_beta^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:K_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_4/DisableCopyOnReadDisableCopyOnRead;read_4_disablecopyonread_batch_normalization_33_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp;read_4_disablecopyonread_batch_normalization_33_moving_mean^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0i

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:K_

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_5/DisableCopyOnReadDisableCopyOnRead?read_5_disablecopyonread_batch_normalization_33_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp?read_5_disablecopyonread_batch_normalization_33_moving_variance^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:K}
Read_6/DisableCopyOnReadDisableCopyOnRead)read_6_disablecopyonread_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp)read_6_disablecopyonread_conv2d_34_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K2*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:K2m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:K2{
Read_7/DisableCopyOnReadDisableCopyOnRead'read_7_disablecopyonread_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp'read_7_disablecopyonread_conv2d_34_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_8/DisableCopyOnReadDisableCopyOnRead5read_8_disablecopyonread_batch_normalization_34_gamma"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp5read_8_disablecopyonread_batch_normalization_34_gamma^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_9/DisableCopyOnReadDisableCopyOnRead4read_9_disablecopyonread_batch_normalization_34_beta"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp4read_9_disablecopyonread_batch_normalization_34_beta^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_10/DisableCopyOnReadDisableCopyOnRead<read_10_disablecopyonread_batch_normalization_34_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp<read_10_disablecopyonread_batch_normalization_34_moving_mean^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_11/DisableCopyOnReadDisableCopyOnRead@read_11_disablecopyonread_batch_normalization_34_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp@read_11_disablecopyonread_batch_normalization_34_moving_variance^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:2
Read_12/DisableCopyOnReadDisableCopyOnRead*read_12_disablecopyonread_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp*read_12_disablecopyonread_conv2d_35_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0w
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2m
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*&
_output_shapes
:2}
Read_13/DisableCopyOnReadDisableCopyOnRead(read_13_disablecopyonread_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp(read_13_disablecopyonread_conv2d_35_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_14/DisableCopyOnReadDisableCopyOnRead6read_14_disablecopyonread_batch_normalization_35_gamma"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp6read_14_disablecopyonread_batch_normalization_35_gamma^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_15/DisableCopyOnReadDisableCopyOnRead5read_15_disablecopyonread_batch_normalization_35_beta"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp5read_15_disablecopyonread_batch_normalization_35_beta^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_16/DisableCopyOnReadDisableCopyOnRead<read_16_disablecopyonread_batch_normalization_35_moving_mean"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp<read_16_disablecopyonread_batch_normalization_35_moving_mean^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_17/DisableCopyOnReadDisableCopyOnRead@read_17_disablecopyonread_batch_normalization_35_moving_variance"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp@read_17_disablecopyonread_batch_normalization_35_moving_variance^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_dense_22_kernel^Read_18/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0* 
_output_shapes
:
��|
Read_19/DisableCopyOnReadDisableCopyOnRead'read_19_disablecopyonread_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp'read_19_disablecopyonread_dense_22_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_dense_23_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�$*
dtype0p
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�$f
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:	�$|
Read_21/DisableCopyOnReadDisableCopyOnRead'read_21_disablecopyonread_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp'read_21_disablecopyonread_dense_23_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:$x
Read_22/DisableCopyOnReadDisableCopyOnRead#read_22_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp#read_22_disablecopyonread_iteration^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_23/DisableCopyOnReadDisableCopyOnRead'read_23_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp'read_23_disablecopyonread_learning_rate^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_24/DisableCopyOnReadDisableCopyOnRead1read_24_disablecopyonread_adam_m_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp1read_24_disablecopyonread_adam_m_conv2d_33_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K*
dtype0w
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Km
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*&
_output_shapes
:K�
Read_25/DisableCopyOnReadDisableCopyOnRead1read_25_disablecopyonread_adam_v_conv2d_33_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp1read_25_disablecopyonread_adam_v_conv2d_33_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K*
dtype0w
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:Km
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*&
_output_shapes
:K�
Read_26/DisableCopyOnReadDisableCopyOnRead/read_26_disablecopyonread_adam_m_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp/read_26_disablecopyonread_adam_m_conv2d_33_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_27/DisableCopyOnReadDisableCopyOnRead/read_27_disablecopyonread_adam_v_conv2d_33_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp/read_27_disablecopyonread_adam_v_conv2d_33_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_28/DisableCopyOnReadDisableCopyOnRead=read_28_disablecopyonread_adam_m_batch_normalization_33_gamma"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp=read_28_disablecopyonread_adam_m_batch_normalization_33_gamma^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_29/DisableCopyOnReadDisableCopyOnRead=read_29_disablecopyonread_adam_v_batch_normalization_33_gamma"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp=read_29_disablecopyonread_adam_v_batch_normalization_33_gamma^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_30/DisableCopyOnReadDisableCopyOnRead<read_30_disablecopyonread_adam_m_batch_normalization_33_beta"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp<read_30_disablecopyonread_adam_m_batch_normalization_33_beta^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_31/DisableCopyOnReadDisableCopyOnRead<read_31_disablecopyonread_adam_v_batch_normalization_33_beta"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp<read_31_disablecopyonread_adam_v_batch_normalization_33_beta^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:K*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:Ka
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:K�
Read_32/DisableCopyOnReadDisableCopyOnRead1read_32_disablecopyonread_adam_m_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp1read_32_disablecopyonread_adam_m_conv2d_34_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K2*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:K2m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:K2�
Read_33/DisableCopyOnReadDisableCopyOnRead1read_33_disablecopyonread_adam_v_conv2d_34_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp1read_33_disablecopyonread_adam_v_conv2d_34_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:K2*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:K2m
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:K2�
Read_34/DisableCopyOnReadDisableCopyOnRead/read_34_disablecopyonread_adam_m_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp/read_34_disablecopyonread_adam_m_conv2d_34_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_35/DisableCopyOnReadDisableCopyOnRead/read_35_disablecopyonread_adam_v_conv2d_34_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp/read_35_disablecopyonread_adam_v_conv2d_34_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_36/DisableCopyOnReadDisableCopyOnRead=read_36_disablecopyonread_adam_m_batch_normalization_34_gamma"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp=read_36_disablecopyonread_adam_m_batch_normalization_34_gamma^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_37/DisableCopyOnReadDisableCopyOnRead=read_37_disablecopyonread_adam_v_batch_normalization_34_gamma"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp=read_37_disablecopyonread_adam_v_batch_normalization_34_gamma^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_38/DisableCopyOnReadDisableCopyOnRead<read_38_disablecopyonread_adam_m_batch_normalization_34_beta"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp<read_38_disablecopyonread_adam_m_batch_normalization_34_beta^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_39/DisableCopyOnReadDisableCopyOnRead<read_39_disablecopyonread_adam_v_batch_normalization_34_beta"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp<read_39_disablecopyonread_adam_v_batch_normalization_34_beta^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:2*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:2a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:2�
Read_40/DisableCopyOnReadDisableCopyOnRead1read_40_disablecopyonread_adam_m_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp1read_40_disablecopyonread_adam_m_conv2d_35_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0w
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2m
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_41/DisableCopyOnReadDisableCopyOnRead1read_41_disablecopyonread_adam_v_conv2d_35_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp1read_41_disablecopyonread_adam_v_conv2d_35_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:2*
dtype0w
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:2m
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*&
_output_shapes
:2�
Read_42/DisableCopyOnReadDisableCopyOnRead/read_42_disablecopyonread_adam_m_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp/read_42_disablecopyonread_adam_m_conv2d_35_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_43/DisableCopyOnReadDisableCopyOnRead/read_43_disablecopyonread_adam_v_conv2d_35_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp/read_43_disablecopyonread_adam_v_conv2d_35_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_44/DisableCopyOnReadDisableCopyOnRead=read_44_disablecopyonread_adam_m_batch_normalization_35_gamma"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp=read_44_disablecopyonread_adam_m_batch_normalization_35_gamma^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_45/DisableCopyOnReadDisableCopyOnRead=read_45_disablecopyonread_adam_v_batch_normalization_35_gamma"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp=read_45_disablecopyonread_adam_v_batch_normalization_35_gamma^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_46/DisableCopyOnReadDisableCopyOnRead<read_46_disablecopyonread_adam_m_batch_normalization_35_beta"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp<read_46_disablecopyonread_adam_m_batch_normalization_35_beta^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_47/DisableCopyOnReadDisableCopyOnRead<read_47_disablecopyonread_adam_v_batch_normalization_35_beta"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp<read_47_disablecopyonread_adam_v_batch_normalization_35_beta^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_48/DisableCopyOnReadDisableCopyOnRead0read_48_disablecopyonread_adam_m_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp0read_48_disablecopyonread_adam_m_dense_22_kernel^Read_48/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_v_dense_22_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_v_dense_22_kernel^Read_49/DisableCopyOnRead"/device:CPU:0* 
_output_shapes
:
��*
dtype0q
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0* 
_output_shapes
:
��g
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0* 
_output_shapes
:
���
Read_50/DisableCopyOnReadDisableCopyOnRead.read_50_disablecopyonread_adam_m_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp.read_50_disablecopyonread_adam_m_dense_22_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead.read_51_disablecopyonread_adam_v_dense_22_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp.read_51_disablecopyonread_adam_v_dense_22_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnRead0read_52_disablecopyonread_adam_m_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp0read_52_disablecopyonread_adam_m_dense_23_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�$*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�$h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	�$�
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_v_dense_23_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_v_dense_23_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�$*
dtype0q
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�$h
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:	�$�
Read_54/DisableCopyOnReadDisableCopyOnRead.read_54_disablecopyonread_adam_m_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp.read_54_disablecopyonread_adam_m_dense_23_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:$�
Read_55/DisableCopyOnReadDisableCopyOnRead.read_55_disablecopyonread_adam_v_dense_23_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp.read_55_disablecopyonread_adam_v_dense_23_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:$*
dtype0l
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:$c
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:$v
Read_56/DisableCopyOnReadDisableCopyOnRead!read_56_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp!read_56_disablecopyonread_total_1^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_57/DisableCopyOnReadDisableCopyOnRead!read_57_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp!read_57_disablecopyonread_count_1^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_58/DisableCopyOnReadDisableCopyOnReadread_58_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOpread_58_disablecopyonread_total^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_59/DisableCopyOnReadDisableCopyOnReadread_59_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOpread_59_disablecopyonread_count^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *K
dtypesA
?2=	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_120Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_121IdentityIdentity_120:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_121Identity_121:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes~
|: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:==9

_output_shapes
: 

_user_specified_nameConst:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:480
.
_user_specified_nameAdam/v/dense_23/bias:470
.
_user_specified_nameAdam/m/dense_23/bias:662
0
_user_specified_nameAdam/v/dense_23/kernel:652
0
_user_specified_nameAdam/m/dense_23/kernel:440
.
_user_specified_nameAdam/v/dense_22/bias:430
.
_user_specified_nameAdam/m/dense_22/bias:622
0
_user_specified_nameAdam/v/dense_22/kernel:612
0
_user_specified_nameAdam/m/dense_22/kernel:B0>
<
_user_specified_name$"Adam/v/batch_normalization_35/beta:B/>
<
_user_specified_name$"Adam/m/batch_normalization_35/beta:C.?
=
_user_specified_name%#Adam/v/batch_normalization_35/gamma:C-?
=
_user_specified_name%#Adam/m/batch_normalization_35/gamma:5,1
/
_user_specified_nameAdam/v/conv2d_35/bias:5+1
/
_user_specified_nameAdam/m/conv2d_35/bias:7*3
1
_user_specified_nameAdam/v/conv2d_35/kernel:7)3
1
_user_specified_nameAdam/m/conv2d_35/kernel:B(>
<
_user_specified_name$"Adam/v/batch_normalization_34/beta:B'>
<
_user_specified_name$"Adam/m/batch_normalization_34/beta:C&?
=
_user_specified_name%#Adam/v/batch_normalization_34/gamma:C%?
=
_user_specified_name%#Adam/m/batch_normalization_34/gamma:5$1
/
_user_specified_nameAdam/v/conv2d_34/bias:5#1
/
_user_specified_nameAdam/m/conv2d_34/bias:7"3
1
_user_specified_nameAdam/v/conv2d_34/kernel:7!3
1
_user_specified_nameAdam/m/conv2d_34/kernel:B >
<
_user_specified_name$"Adam/v/batch_normalization_33/beta:B>
<
_user_specified_name$"Adam/m/batch_normalization_33/beta:C?
=
_user_specified_name%#Adam/v/batch_normalization_33/gamma:C?
=
_user_specified_name%#Adam/m/batch_normalization_33/gamma:51
/
_user_specified_nameAdam/v/conv2d_33/bias:51
/
_user_specified_nameAdam/m/conv2d_33/bias:73
1
_user_specified_nameAdam/v/conv2d_33/kernel:73
1
_user_specified_nameAdam/m/conv2d_33/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_23/bias:/+
)
_user_specified_namedense_23/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:FB
@
_user_specified_name(&batch_normalization_35/moving_variance:B>
<
_user_specified_name$"batch_normalization_35/moving_mean:;7
5
_user_specified_namebatch_normalization_35/beta:<8
6
_user_specified_namebatch_normalization_35/gamma:.*
(
_user_specified_nameconv2d_35/bias:0,
*
_user_specified_nameconv2d_35/kernel:FB
@
_user_specified_name(&batch_normalization_34/moving_variance:B>
<
_user_specified_name$"batch_normalization_34/moving_mean:;
7
5
_user_specified_namebatch_normalization_34/beta:<	8
6
_user_specified_namebatch_normalization_34/gamma:.*
(
_user_specified_nameconv2d_34/bias:0,
*
_user_specified_nameconv2d_34/kernel:FB
@
_user_specified_name(&batch_normalization_33/moving_variance:B>
<
_user_specified_name$"batch_normalization_33/moving_mean:;7
5
_user_specified_namebatch_normalization_33/beta:<8
6
_user_specified_namebatch_normalization_33/gamma:.*
(
_user_specified_nameconv2d_33/bias:0,
*
_user_specified_nameconv2d_33/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159176

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2$
AssignNewValue_1AssignNewValue_12 
AssignNewValueAssignNewValue2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
b
F__inference_flatten_11_layer_call_and_return_conditional_losses_158561

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
ޚ
�'
"__inference__traced_restore_159853
file_prefix;
!assignvariableop_conv2d_33_kernel:K/
!assignvariableop_1_conv2d_33_bias:K=
/assignvariableop_2_batch_normalization_33_gamma:K<
.assignvariableop_3_batch_normalization_33_beta:KC
5assignvariableop_4_batch_normalization_33_moving_mean:KG
9assignvariableop_5_batch_normalization_33_moving_variance:K=
#assignvariableop_6_conv2d_34_kernel:K2/
!assignvariableop_7_conv2d_34_bias:2=
/assignvariableop_8_batch_normalization_34_gamma:2<
.assignvariableop_9_batch_normalization_34_beta:2D
6assignvariableop_10_batch_normalization_34_moving_mean:2H
:assignvariableop_11_batch_normalization_34_moving_variance:2>
$assignvariableop_12_conv2d_35_kernel:20
"assignvariableop_13_conv2d_35_bias:>
0assignvariableop_14_batch_normalization_35_gamma:=
/assignvariableop_15_batch_normalization_35_beta:D
6assignvariableop_16_batch_normalization_35_moving_mean:H
:assignvariableop_17_batch_normalization_35_moving_variance:7
#assignvariableop_18_dense_22_kernel:
��0
!assignvariableop_19_dense_22_bias:	�6
#assignvariableop_20_dense_23_kernel:	�$/
!assignvariableop_21_dense_23_bias:$'
assignvariableop_22_iteration:	 +
!assignvariableop_23_learning_rate: E
+assignvariableop_24_adam_m_conv2d_33_kernel:KE
+assignvariableop_25_adam_v_conv2d_33_kernel:K7
)assignvariableop_26_adam_m_conv2d_33_bias:K7
)assignvariableop_27_adam_v_conv2d_33_bias:KE
7assignvariableop_28_adam_m_batch_normalization_33_gamma:KE
7assignvariableop_29_adam_v_batch_normalization_33_gamma:KD
6assignvariableop_30_adam_m_batch_normalization_33_beta:KD
6assignvariableop_31_adam_v_batch_normalization_33_beta:KE
+assignvariableop_32_adam_m_conv2d_34_kernel:K2E
+assignvariableop_33_adam_v_conv2d_34_kernel:K27
)assignvariableop_34_adam_m_conv2d_34_bias:27
)assignvariableop_35_adam_v_conv2d_34_bias:2E
7assignvariableop_36_adam_m_batch_normalization_34_gamma:2E
7assignvariableop_37_adam_v_batch_normalization_34_gamma:2D
6assignvariableop_38_adam_m_batch_normalization_34_beta:2D
6assignvariableop_39_adam_v_batch_normalization_34_beta:2E
+assignvariableop_40_adam_m_conv2d_35_kernel:2E
+assignvariableop_41_adam_v_conv2d_35_kernel:27
)assignvariableop_42_adam_m_conv2d_35_bias:7
)assignvariableop_43_adam_v_conv2d_35_bias:E
7assignvariableop_44_adam_m_batch_normalization_35_gamma:E
7assignvariableop_45_adam_v_batch_normalization_35_gamma:D
6assignvariableop_46_adam_m_batch_normalization_35_beta:D
6assignvariableop_47_adam_v_batch_normalization_35_beta:>
*assignvariableop_48_adam_m_dense_22_kernel:
��>
*assignvariableop_49_adam_v_dense_22_kernel:
��7
(assignvariableop_50_adam_m_dense_22_bias:	�7
(assignvariableop_51_adam_v_dense_22_bias:	�=
*assignvariableop_52_adam_m_dense_23_kernel:	�$=
*assignvariableop_53_adam_v_dense_23_kernel:	�$6
(assignvariableop_54_adam_m_dense_23_bias:$6
(assignvariableop_55_adam_v_dense_23_bias:$%
assignvariableop_56_total_1: %
assignvariableop_57_count_1: #
assignvariableop_58_total: #
assignvariableop_59_count: 
identity_61��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-1/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-1/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-3/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-3/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:=*
dtype0*�
value�B�=B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*K
dtypesA
?2=	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_33_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_33_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp/assignvariableop_2_batch_normalization_33_gammaIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp.assignvariableop_3_batch_normalization_33_betaIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp5assignvariableop_4_batch_normalization_33_moving_meanIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp9assignvariableop_5_batch_normalization_33_moving_varianceIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_34_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_34_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp/assignvariableop_8_batch_normalization_34_gammaIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp.assignvariableop_9_batch_normalization_34_betaIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp6assignvariableop_10_batch_normalization_34_moving_meanIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp:assignvariableop_11_batch_normalization_34_moving_varianceIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp$assignvariableop_12_conv2d_35_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp"assignvariableop_13_conv2d_35_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp0assignvariableop_14_batch_normalization_35_gammaIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp/assignvariableop_15_batch_normalization_35_betaIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp6assignvariableop_16_batch_normalization_35_moving_meanIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp:assignvariableop_17_batch_normalization_35_moving_varianceIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp#assignvariableop_18_dense_22_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp!assignvariableop_19_dense_22_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp#assignvariableop_20_dense_23_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp!assignvariableop_21_dense_23_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_22AssignVariableOpassignvariableop_22_iterationIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp!assignvariableop_23_learning_rateIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_m_conv2d_33_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp+assignvariableop_25_adam_v_conv2d_33_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_m_conv2d_33_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_v_conv2d_33_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp7assignvariableop_28_adam_m_batch_normalization_33_gammaIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_v_batch_normalization_33_gammaIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_30AssignVariableOp6assignvariableop_30_adam_m_batch_normalization_33_betaIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp6assignvariableop_31_adam_v_batch_normalization_33_betaIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_m_conv2d_34_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_v_conv2d_34_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_m_conv2d_34_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_v_conv2d_34_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp7assignvariableop_36_adam_m_batch_normalization_34_gammaIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_v_batch_normalization_34_gammaIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp6assignvariableop_38_adam_m_batch_normalization_34_betaIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp6assignvariableop_39_adam_v_batch_normalization_34_betaIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp+assignvariableop_40_adam_m_conv2d_35_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_v_conv2d_35_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_m_conv2d_35_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_v_conv2d_35_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp7assignvariableop_44_adam_m_batch_normalization_35_gammaIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_v_batch_normalization_35_gammaIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_m_batch_normalization_35_betaIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp6assignvariableop_47_adam_v_batch_normalization_35_betaIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp*assignvariableop_48_adam_m_dense_22_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_v_dense_22_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp(assignvariableop_50_adam_m_dense_22_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp(assignvariableop_51_adam_v_dense_22_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp*assignvariableop_52_adam_m_dense_23_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_v_dense_23_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_m_dense_23_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp(assignvariableop_55_adam_v_dense_23_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOpassignvariableop_56_total_1Identity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOpassignvariableop_57_count_1Identity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOpassignvariableop_58_totalIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOpassignvariableop_59_countIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_60Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_61IdentityIdentity_60:output:0^NoOp_1*
T0*
_output_shapes
: �

NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_61Identity_61:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes|
z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%<!

_user_specified_namecount:%;!

_user_specified_nametotal:':#
!
_user_specified_name	count_1:'9#
!
_user_specified_name	total_1:480
.
_user_specified_nameAdam/v/dense_23/bias:470
.
_user_specified_nameAdam/m/dense_23/bias:662
0
_user_specified_nameAdam/v/dense_23/kernel:652
0
_user_specified_nameAdam/m/dense_23/kernel:440
.
_user_specified_nameAdam/v/dense_22/bias:430
.
_user_specified_nameAdam/m/dense_22/bias:622
0
_user_specified_nameAdam/v/dense_22/kernel:612
0
_user_specified_nameAdam/m/dense_22/kernel:B0>
<
_user_specified_name$"Adam/v/batch_normalization_35/beta:B/>
<
_user_specified_name$"Adam/m/batch_normalization_35/beta:C.?
=
_user_specified_name%#Adam/v/batch_normalization_35/gamma:C-?
=
_user_specified_name%#Adam/m/batch_normalization_35/gamma:5,1
/
_user_specified_nameAdam/v/conv2d_35/bias:5+1
/
_user_specified_nameAdam/m/conv2d_35/bias:7*3
1
_user_specified_nameAdam/v/conv2d_35/kernel:7)3
1
_user_specified_nameAdam/m/conv2d_35/kernel:B(>
<
_user_specified_name$"Adam/v/batch_normalization_34/beta:B'>
<
_user_specified_name$"Adam/m/batch_normalization_34/beta:C&?
=
_user_specified_name%#Adam/v/batch_normalization_34/gamma:C%?
=
_user_specified_name%#Adam/m/batch_normalization_34/gamma:5$1
/
_user_specified_nameAdam/v/conv2d_34/bias:5#1
/
_user_specified_nameAdam/m/conv2d_34/bias:7"3
1
_user_specified_nameAdam/v/conv2d_34/kernel:7!3
1
_user_specified_nameAdam/m/conv2d_34/kernel:B >
<
_user_specified_name$"Adam/v/batch_normalization_33/beta:B>
<
_user_specified_name$"Adam/m/batch_normalization_33/beta:C?
=
_user_specified_name%#Adam/v/batch_normalization_33/gamma:C?
=
_user_specified_name%#Adam/m/batch_normalization_33/gamma:51
/
_user_specified_nameAdam/v/conv2d_33/bias:51
/
_user_specified_nameAdam/m/conv2d_33/bias:73
1
_user_specified_nameAdam/v/conv2d_33/kernel:73
1
_user_specified_nameAdam/m/conv2d_33/kernel:-)
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:-)
'
_user_specified_namedense_23/bias:/+
)
_user_specified_namedense_23/kernel:-)
'
_user_specified_namedense_22/bias:/+
)
_user_specified_namedense_22/kernel:FB
@
_user_specified_name(&batch_normalization_35/moving_variance:B>
<
_user_specified_name$"batch_normalization_35/moving_mean:;7
5
_user_specified_namebatch_normalization_35/beta:<8
6
_user_specified_namebatch_normalization_35/gamma:.*
(
_user_specified_nameconv2d_35/bias:0,
*
_user_specified_nameconv2d_35/kernel:FB
@
_user_specified_name(&batch_normalization_34/moving_variance:B>
<
_user_specified_name$"batch_normalization_34/moving_mean:;
7
5
_user_specified_namebatch_normalization_34/beta:<	8
6
_user_specified_namebatch_normalization_34/gamma:.*
(
_user_specified_nameconv2d_34/bias:0,
*
_user_specified_nameconv2d_34/kernel:FB
@
_user_specified_name(&batch_normalization_33/moving_variance:B>
<
_user_specified_name$"batch_normalization_33/moving_mean:;7
5
_user_specified_namebatch_normalization_33/beta:<8
6
_user_specified_namebatch_normalization_33/gamma:.*
(
_user_specified_nameconv2d_33/bias:0,
*
_user_specified_nameconv2d_33/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
h
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_159112

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingSAME*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

e
F__inference_dropout_22_layer_call_and_return_conditional_losses_159035

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  �?l
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:���������  2Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:���������  2*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *��L>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:���������  2T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*/
_output_shapes
:���������  2i
IdentityIdentitydropout/SelectV2:output:0*
T0*/
_output_shapes
:���������  2"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  2:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�

�
7__inference_batch_normalization_35_layer_call_fn_159158

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158426�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:&"
 
_user_specified_name159154:&"
 
_user_specified_name159152:&"
 
_user_specified_name159150:&"
 
_user_specified_name159148:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158426

inputs%
readvariableop_resource:'
readvariableop_1_resource:6
(fusedbatchnormv3_readvariableop_resource:8
*fusedbatchnormv3_readvariableop_1_resource:
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������:::::*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+����������������������������
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������
 
_user_specified_nameinputs
�

e
F__inference_dropout_23_layer_call_and_return_conditional_losses_158590

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�M
�
I__inference_sequential_11_layer_call_and_return_conditional_losses_158609
conv2d_33_input*
conv2d_33_158476:K
conv2d_33_158478:K+
batch_normalization_33_158481:K+
batch_normalization_33_158483:K+
batch_normalization_33_158485:K+
batch_normalization_33_158487:K*
conv2d_34_158502:K2
conv2d_34_158504:2+
batch_normalization_34_158520:2+
batch_normalization_34_158522:2+
batch_normalization_34_158524:2+
batch_normalization_34_158526:2*
conv2d_35_158541:2
conv2d_35_158543:+
batch_normalization_35_158546:+
batch_normalization_35_158548:+
batch_normalization_35_158550:+
batch_normalization_35_158552:#
dense_22_158574:
��
dense_22_158576:	�"
dense_23_158603:	�$
dense_23_158605:$
identity��.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�"dropout_22/StatefulPartitionedCall�"dropout_23/StatefulPartitionedCall�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCallconv2d_33_inputconv2d_33_158476conv2d_33_158478*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158475�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_33_158481batch_normalization_33_158483batch_normalization_33_158485batch_normalization_33_158487*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158264�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  K* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158313�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_34_158502conv2d_34_158504*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_158501�
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_158518�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0batch_normalization_34_158520batch_normalization_34_158522batch_normalization_34_158524batch_normalization_34_158526*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158336�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_158385�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_35_158541conv2d_35_158543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_158540�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_158546batch_normalization_35_158548batch_normalization_35_158550batch_normalization_35_158552*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158408�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_158457�
flatten_11/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_158561�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_158574dense_22_158576*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_158573�
"dropout_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_22/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_158590�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_23/StatefulPartitionedCall:output:0dense_23_158603dense_23_158605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_158602x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall#^dropout_23/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall2H
"dropout_23/StatefulPartitionedCall"dropout_23/StatefulPartitionedCall:&"
 
_user_specified_name158605:&"
 
_user_specified_name158603:&"
 
_user_specified_name158576:&"
 
_user_specified_name158574:&"
 
_user_specified_name158552:&"
 
_user_specified_name158550:&"
 
_user_specified_name158548:&"
 
_user_specified_name158546:&"
 
_user_specified_name158543:&"
 
_user_specified_name158541:&"
 
_user_specified_name158526:&"
 
_user_specified_name158524:&
"
 
_user_specified_name158522:&	"
 
_user_specified_name158520:&"
 
_user_specified_name158504:&"
 
_user_specified_name158502:&"
 
_user_specified_name158487:&"
 
_user_specified_name158485:&"
 
_user_specified_name158483:&"
 
_user_specified_name158481:&"
 
_user_specified_name158478:&"
 
_user_specified_name158476:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�
�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158282

inputs%
readvariableop_resource:K'
readvariableop_1_resource:K6
(fusedbatchnormv3_readvariableop_resource:K8
*fusedbatchnormv3_readvariableop_1_resource:K
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:K*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:K*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������K:K:K:K:K:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�
M
1__inference_max_pooling2d_34_layer_call_fn_159107

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_158385�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158983

inputs%
readvariableop_resource:K'
readvariableop_1_resource:K6
(fusedbatchnormv3_readvariableop_resource:K8
*fusedbatchnormv3_readvariableop_1_resource:K
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:K*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:K*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:K*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������K:K:K:K:K:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������K�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������K: : : : 2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12 
ReadVariableOpReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:i e
A
_output_shapes/
-:+���������������������������K
 
_user_specified_nameinputs
�

�
D__inference_dense_23_layer_call_and_return_conditional_losses_158602

inputs1
matmul_readvariableop_resource:	�$-
biasadd_readvariableop_resource:$
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�$*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������$`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������$S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_158636

inputs

identity_1V
IdentityIdentityinputs*
T0*/
_output_shapes
:���������  2c

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:���������  2"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������  2:W S
/
_output_shapes
:���������  2
 
_user_specified_nameinputs
�J
�

I__inference_sequential_11_layer_call_and_return_conditional_losses_158681
conv2d_33_input*
conv2d_33_158612:K
conv2d_33_158614:K+
batch_normalization_33_158617:K+
batch_normalization_33_158619:K+
batch_normalization_33_158621:K+
batch_normalization_33_158623:K*
conv2d_34_158627:K2
conv2d_34_158629:2+
batch_normalization_34_158638:2+
batch_normalization_34_158640:2+
batch_normalization_34_158642:2+
batch_normalization_34_158644:2*
conv2d_35_158648:2
conv2d_35_158650:+
batch_normalization_35_158653:+
batch_normalization_35_158655:+
batch_normalization_35_158657:+
batch_normalization_35_158659:#
dense_22_158664:
��
dense_22_158666:	�"
dense_23_158675:	�$
dense_23_158677:$
identity��.batch_normalization_33/StatefulPartitionedCall�.batch_normalization_34/StatefulPartitionedCall�.batch_normalization_35/StatefulPartitionedCall�!conv2d_33/StatefulPartitionedCall�!conv2d_34/StatefulPartitionedCall�!conv2d_35/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall�
!conv2d_33/StatefulPartitionedCallStatefulPartitionedCallconv2d_33_inputconv2d_33_158612conv2d_33_158614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@K*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158475�
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall*conv2d_33/StatefulPartitionedCall:output:0batch_normalization_33_158617batch_normalization_33_158619batch_normalization_33_158621batch_normalization_33_158623*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������@@K*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158282�
 max_pooling2d_33/PartitionedCallPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  K* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158313�
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0conv2d_34_158627conv2d_34_158629*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_34_layer_call_and_return_conditional_losses_158501�
dropout_22/PartitionedCallPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_158636�
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0batch_normalization_34_158638batch_normalization_34_158640batch_normalization_34_158642batch_normalization_34_158644*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������  2*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_158354�
 max_pooling2d_34/PartitionedCallPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������2* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_158385�
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0conv2d_35_158648conv2d_35_158650*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_conv2d_35_layer_call_and_return_conditional_losses_158540�
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0batch_normalization_35_158653batch_normalization_35_158655batch_normalization_35_158657batch_normalization_35_158659*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *[
fVRT
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_158426�
 max_pooling2d_35/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *U
fPRN
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_158457�
flatten_11/PartitionedCallPartitionedCall)max_pooling2d_35/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_158561�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_158664dense_22_158666*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_158573�
dropout_23/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_158673�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_23/PartitionedCall:output:0dense_23_158675dense_23_158677*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_158602x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������$�
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_33/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Z
_input_shapesI
G:���������@@: : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_33/StatefulPartitionedCall!conv2d_33/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:&"
 
_user_specified_name158677:&"
 
_user_specified_name158675:&"
 
_user_specified_name158666:&"
 
_user_specified_name158664:&"
 
_user_specified_name158659:&"
 
_user_specified_name158657:&"
 
_user_specified_name158655:&"
 
_user_specified_name158653:&"
 
_user_specified_name158650:&"
 
_user_specified_name158648:&"
 
_user_specified_name158644:&"
 
_user_specified_name158642:&
"
 
_user_specified_name158640:&	"
 
_user_specified_name158638:&"
 
_user_specified_name158629:&"
 
_user_specified_name158627:&"
 
_user_specified_name158623:&"
 
_user_specified_name158621:&"
 
_user_specified_name158619:&"
 
_user_specified_name158617:&"
 
_user_specified_name158614:&"
 
_user_specified_name158612:` \
/
_output_shapes
:���������@@
)
_user_specified_nameconv2d_33_input
�
G
+__inference_flatten_11_layer_call_fn_159209

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_flatten_11_layer_call_and_return_conditional_losses_158561a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������:W S
/
_output_shapes
:���������
 
_user_specified_nameinputs
�
d
F__inference_dropout_23_layer_call_and_return_conditional_losses_158673

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:����������\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
d
+__inference_dropout_23_layer_call_fn_159240

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dropout_23_layer_call_and_return_conditional_losses_158590p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:����������22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
S
conv2d_33_input@
!serving_default_conv2d_33_input:0���������@@<
dense_230
StatefulPartitionedCall:0���������$tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer_with_weights-5
	layer-8

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op"
_tf_keras_layer
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses
'axis
	(gamma
)beta
*moving_mean
+moving_variance"
_tf_keras_layer
�
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

8kernel
9bias
 :_jit_compiled_convolution_op"
_tf_keras_layer
�
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A_random_generator"
_tf_keras_layer
�
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
Haxis
	Igamma
Jbeta
Kmoving_mean
Lmoving_variance"
_tf_keras_layer
�
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
�
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias
 [_jit_compiled_convolution_op"
_tf_keras_layer
�
\	variables
]trainable_variables
^regularization_losses
_	keras_api
`__call__
*a&call_and_return_all_conditional_losses
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses"
_tf_keras_layer
�
s	variables
ttrainable_variables
uregularization_losses
v	keras_api
w__call__
*x&call_and_return_all_conditional_losses

ykernel
zbias"
_tf_keras_layer
�
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
0
1
(2
)3
*4
+5
86
97
I8
J9
K10
L11
Y12
Z13
c14
d15
e16
f17
y18
z19
�20
�21"
trackable_list_wrapper
�
0
1
(2
)3
84
95
I6
J7
Y8
Z9
c10
d11
y12
z13
�14
�15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
.__inference_sequential_11_layer_call_fn_158730
.__inference_sequential_11_layer_call_fn_158779�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
I__inference_sequential_11_layer_call_and_return_conditional_losses_158609
I__inference_sequential_11_layer_call_and_return_conditional_losses_158681�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
!__inference__wrapped_model_158246conv2d_33_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�
_variables
�_iterations
�_learning_rate
�_index_dict
�
_momentums
�_velocities
�_update_step_xla"
experimentalOptimizer
-
�serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_33_layer_call_fn_158910�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158921�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(K2conv2d_33/kernel
:K2conv2d_33/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
(0
)1
*2
+3"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_33_layer_call_fn_158934
7__inference_batch_normalization_33_layer_call_fn_158947�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158965
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158983�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(K2batch_normalization_33/gamma
):'K2batch_normalization_33/beta
2:0K (2"batch_normalization_33/moving_mean
6:4K (2&batch_normalization_33/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_33_layer_call_fn_158988�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158993�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_34_layer_call_fn_159002�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_34_layer_call_and_return_conditional_losses_159013�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(K22conv2d_34/kernel
:22conv2d_34/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_22_layer_call_fn_159018
+__inference_dropout_22_layer_call_fn_159023�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_22_layer_call_and_return_conditional_losses_159035
F__inference_dropout_22_layer_call_and_return_conditional_losses_159040�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
<
I0
J1
K2
L3"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_34_layer_call_fn_159053
7__inference_batch_normalization_34_layer_call_fn_159066�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159084
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159102�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(22batch_normalization_34/gamma
):'22batch_normalization_34/beta
2:02 (2"batch_normalization_34/moving_mean
6:42 (2&batch_normalization_34/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_34_layer_call_fn_159107�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_159112�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_conv2d_35_layer_call_fn_159121�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_conv2d_35_layer_call_and_return_conditional_losses_159132�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
*:(22conv2d_35/kernel
:2conv2d_35/bias
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
<
c0
d1
e2
f3"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
\	variables
]trainable_variables
^regularization_losses
`__call__
*a&call_and_return_all_conditional_losses
&a"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_batch_normalization_35_layer_call_fn_159145
7__inference_batch_normalization_35_layer_call_fn_159158�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159176
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159194�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
*:(2batch_normalization_35/gamma
):'2batch_normalization_35/beta
2:0 (2"batch_normalization_35/moving_mean
6:4 (2&batch_normalization_35/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_max_pooling2d_35_layer_call_fn_159199�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_159204�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_flatten_11_layer_call_fn_159209�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_flatten_11_layer_call_and_return_conditional_losses_159215�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
y0
z1"
trackable_list_wrapper
.
y0
z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
s	variables
ttrainable_variables
uregularization_losses
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_22_layer_call_fn_159224�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_22_layer_call_and_return_conditional_losses_159235�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
#:!
��2dense_22/kernel
:�2dense_22/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
+__inference_dropout_23_layer_call_fn_159240
+__inference_dropout_23_layer_call_fn_159245�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
F__inference_dropout_23_layer_call_and_return_conditional_losses_159257
F__inference_dropout_23_layer_call_and_return_conditional_losses_159262�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
)__inference_dense_23_layer_call_fn_159271�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
D__inference_dense_23_layer_call_and_return_conditional_losses_159282�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": 	�$2dense_23/kernel
:$2dense_23/bias
J
*0
+1
K2
L3
e4
f5"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
.__inference_sequential_11_layer_call_fn_158730conv2d_33_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
.__inference_sequential_11_layer_call_fn_158779conv2d_33_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_11_layer_call_and_return_conditional_losses_158609conv2d_33_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_sequential_11_layer_call_and_return_conditional_losses_158681conv2d_33_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
$__inference_signature_wrapper_158901conv2d_33_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 $

kwonlyargs�
jconv2d_33_input
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_33_layer_call_fn_158910inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158921inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_33_layer_call_fn_158934inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_33_layer_call_fn_158947inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158965inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158983inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_33_layer_call_fn_158988inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158993inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_34_layer_call_fn_159002inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_34_layer_call_and_return_conditional_losses_159013inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_22_layer_call_fn_159018inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_22_layer_call_fn_159023inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_22_layer_call_and_return_conditional_losses_159035inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_22_layer_call_and_return_conditional_losses_159040inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_34_layer_call_fn_159053inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_34_layer_call_fn_159066inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159084inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159102inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_34_layer_call_fn_159107inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_159112inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_conv2d_35_layer_call_fn_159121inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_conv2d_35_layer_call_and_return_conditional_losses_159132inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_batch_normalization_35_layer_call_fn_159145inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_batch_normalization_35_layer_call_fn_159158inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159176inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159194inputs"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
1__inference_max_pooling2d_35_layer_call_fn_159199inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_159204inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_flatten_11_layer_call_fn_159209inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_flatten_11_layer_call_and_return_conditional_losses_159215inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_22_layer_call_fn_159224inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_22_layer_call_and_return_conditional_losses_159235inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dropout_23_layer_call_fn_159240inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
+__inference_dropout_23_layer_call_fn_159245inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_23_layer_call_and_return_conditional_losses_159257inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dropout_23_layer_call_and_return_conditional_losses_159262inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_dense_23_layer_call_fn_159271inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_dense_23_layer_call_and_return_conditional_losses_159282inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
/:-K2Adam/m/conv2d_33/kernel
/:-K2Adam/v/conv2d_33/kernel
!:K2Adam/m/conv2d_33/bias
!:K2Adam/v/conv2d_33/bias
/:-K2#Adam/m/batch_normalization_33/gamma
/:-K2#Adam/v/batch_normalization_33/gamma
.:,K2"Adam/m/batch_normalization_33/beta
.:,K2"Adam/v/batch_normalization_33/beta
/:-K22Adam/m/conv2d_34/kernel
/:-K22Adam/v/conv2d_34/kernel
!:22Adam/m/conv2d_34/bias
!:22Adam/v/conv2d_34/bias
/:-22#Adam/m/batch_normalization_34/gamma
/:-22#Adam/v/batch_normalization_34/gamma
.:,22"Adam/m/batch_normalization_34/beta
.:,22"Adam/v/batch_normalization_34/beta
/:-22Adam/m/conv2d_35/kernel
/:-22Adam/v/conv2d_35/kernel
!:2Adam/m/conv2d_35/bias
!:2Adam/v/conv2d_35/bias
/:-2#Adam/m/batch_normalization_35/gamma
/:-2#Adam/v/batch_normalization_35/gamma
.:,2"Adam/m/batch_normalization_35/beta
.:,2"Adam/v/batch_normalization_35/beta
(:&
��2Adam/m/dense_22/kernel
(:&
��2Adam/v/dense_22/kernel
!:�2Adam/m/dense_22/bias
!:�2Adam/v/dense_22/bias
':%	�$2Adam/m/dense_23/kernel
':%	�$2Adam/v/dense_23/kernel
 :$2Adam/m/dense_23/bias
 :$2Adam/v/dense_23/bias
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
!__inference__wrapped_model_158246�()*+89IJKLYZcdefyz��@�=
6�3
1�.
conv2d_33_input���������@@
� "3�0
.
dense_23"�
dense_23���������$�
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158965�()*+Q�N
G�D
:�7
inputs+���������������������������K
p

 
� "F�C
<�9
tensor_0+���������������������������K
� �
R__inference_batch_normalization_33_layer_call_and_return_conditional_losses_158983�()*+Q�N
G�D
:�7
inputs+���������������������������K
p 

 
� "F�C
<�9
tensor_0+���������������������������K
� �
7__inference_batch_normalization_33_layer_call_fn_158934�()*+Q�N
G�D
:�7
inputs+���������������������������K
p

 
� ";�8
unknown+���������������������������K�
7__inference_batch_normalization_33_layer_call_fn_158947�()*+Q�N
G�D
:�7
inputs+���������������������������K
p 

 
� ";�8
unknown+���������������������������K�
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159084�IJKLQ�N
G�D
:�7
inputs+���������������������������2
p

 
� "F�C
<�9
tensor_0+���������������������������2
� �
R__inference_batch_normalization_34_layer_call_and_return_conditional_losses_159102�IJKLQ�N
G�D
:�7
inputs+���������������������������2
p 

 
� "F�C
<�9
tensor_0+���������������������������2
� �
7__inference_batch_normalization_34_layer_call_fn_159053�IJKLQ�N
G�D
:�7
inputs+���������������������������2
p

 
� ";�8
unknown+���������������������������2�
7__inference_batch_normalization_34_layer_call_fn_159066�IJKLQ�N
G�D
:�7
inputs+���������������������������2
p 

 
� ";�8
unknown+���������������������������2�
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159176�cdefQ�N
G�D
:�7
inputs+���������������������������
p

 
� "F�C
<�9
tensor_0+���������������������������
� �
R__inference_batch_normalization_35_layer_call_and_return_conditional_losses_159194�cdefQ�N
G�D
:�7
inputs+���������������������������
p 

 
� "F�C
<�9
tensor_0+���������������������������
� �
7__inference_batch_normalization_35_layer_call_fn_159145�cdefQ�N
G�D
:�7
inputs+���������������������������
p

 
� ";�8
unknown+����������������������������
7__inference_batch_normalization_35_layer_call_fn_159158�cdefQ�N
G�D
:�7
inputs+���������������������������
p 

 
� ";�8
unknown+����������������������������
E__inference_conv2d_33_layer_call_and_return_conditional_losses_158921s7�4
-�*
(�%
inputs���������@@
� "4�1
*�'
tensor_0���������@@K
� �
*__inference_conv2d_33_layer_call_fn_158910h7�4
-�*
(�%
inputs���������@@
� ")�&
unknown���������@@K�
E__inference_conv2d_34_layer_call_and_return_conditional_losses_159013s897�4
-�*
(�%
inputs���������  K
� "4�1
*�'
tensor_0���������  2
� �
*__inference_conv2d_34_layer_call_fn_159002h897�4
-�*
(�%
inputs���������  K
� ")�&
unknown���������  2�
E__inference_conv2d_35_layer_call_and_return_conditional_losses_159132sYZ7�4
-�*
(�%
inputs���������2
� "4�1
*�'
tensor_0���������
� �
*__inference_conv2d_35_layer_call_fn_159121hYZ7�4
-�*
(�%
inputs���������2
� ")�&
unknown����������
D__inference_dense_22_layer_call_and_return_conditional_losses_159235eyz0�-
&�#
!�
inputs����������
� "-�*
#� 
tensor_0����������
� �
)__inference_dense_22_layer_call_fn_159224Zyz0�-
&�#
!�
inputs����������
� ""�
unknown�����������
D__inference_dense_23_layer_call_and_return_conditional_losses_159282f��0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������$
� �
)__inference_dense_23_layer_call_fn_159271[��0�-
&�#
!�
inputs����������
� "!�
unknown���������$�
F__inference_dropout_22_layer_call_and_return_conditional_losses_159035s;�8
1�.
(�%
inputs���������  2
p
� "4�1
*�'
tensor_0���������  2
� �
F__inference_dropout_22_layer_call_and_return_conditional_losses_159040s;�8
1�.
(�%
inputs���������  2
p 
� "4�1
*�'
tensor_0���������  2
� �
+__inference_dropout_22_layer_call_fn_159018h;�8
1�.
(�%
inputs���������  2
p
� ")�&
unknown���������  2�
+__inference_dropout_22_layer_call_fn_159023h;�8
1�.
(�%
inputs���������  2
p 
� ")�&
unknown���������  2�
F__inference_dropout_23_layer_call_and_return_conditional_losses_159257e4�1
*�'
!�
inputs����������
p
� "-�*
#� 
tensor_0����������
� �
F__inference_dropout_23_layer_call_and_return_conditional_losses_159262e4�1
*�'
!�
inputs����������
p 
� "-�*
#� 
tensor_0����������
� �
+__inference_dropout_23_layer_call_fn_159240Z4�1
*�'
!�
inputs����������
p
� ""�
unknown�����������
+__inference_dropout_23_layer_call_fn_159245Z4�1
*�'
!�
inputs����������
p 
� ""�
unknown�����������
F__inference_flatten_11_layer_call_and_return_conditional_losses_159215h7�4
-�*
(�%
inputs���������
� "-�*
#� 
tensor_0����������
� �
+__inference_flatten_11_layer_call_fn_159209]7�4
-�*
(�%
inputs���������
� ""�
unknown�����������
L__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_158993�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_33_layer_call_fn_158988�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_159112�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_34_layer_call_fn_159107�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
L__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_159204�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
1__inference_max_pooling2d_35_layer_call_fn_159199�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
I__inference_sequential_11_layer_call_and_return_conditional_losses_158609�()*+89IJKLYZcdefyz��H�E
>�;
1�.
conv2d_33_input���������@@
p

 
� ",�)
"�
tensor_0���������$
� �
I__inference_sequential_11_layer_call_and_return_conditional_losses_158681�()*+89IJKLYZcdefyz��H�E
>�;
1�.
conv2d_33_input���������@@
p 

 
� ",�)
"�
tensor_0���������$
� �
.__inference_sequential_11_layer_call_fn_158730�()*+89IJKLYZcdefyz��H�E
>�;
1�.
conv2d_33_input���������@@
p

 
� "!�
unknown���������$�
.__inference_sequential_11_layer_call_fn_158779�()*+89IJKLYZcdefyz��H�E
>�;
1�.
conv2d_33_input���������@@
p 

 
� "!�
unknown���������$�
$__inference_signature_wrapper_158901�()*+89IJKLYZcdefyz��S�P
� 
I�F
D
conv2d_33_input1�.
conv2d_33_input���������@@"3�0
.
dense_23"�
dense_23���������$