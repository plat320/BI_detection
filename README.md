# BI_detection
BI_detection

argparser 설명

parser.add_argument('-b', '--batch_size', type=int, default=4, metavar='N', help='batch size for data loader')
parser.add_argument('-e', '--num_epochs', type=int, default=100, metavar='N', help='# of training epoch')
parser.add_argument('-l', '--init_lr', type=float, default=1e-5, metavar='N', help='initial learning rate')
parser.add_argument('-d', '--dataset', type=str, default="mobticon", metavar='N', help='mobticon')
parser.add_argument('--num_classes', type=int, default=4, help='the # of classes')
parser.add_argument('--OOD_num_classes', type=int, default=0, help='the # of OOD classes, if do not want training transfer, this value must be 0')
parser.add_argument('-m','--net_type', type=str, required=True, help='resnet34 | resnet50 | vgg16 | vgg16_bn | vgg19 | vgg19_bn')
parser.add_argument('--where', type=str, default="local", help='which machine')
parser.add_argument('--gpu', type=str, default=0, help='gpu index')
parser.add_argument('--same_class', type=str, nargs="+", default=[], help='same classes\' number')
parser.add_argument('--except_class', type=str, nargs="+", default=[], help='except classes\' number')
parser.add_argument('--OOD_class', type=str, nargs="+", default=[], help='OOD classes\' number')
parser.add_argument('--resume', default=False, action="store_true", help='load last checkpoint')
parser.add_argument('--board_clear', default=False, action="store_true", help='clear tensorboard folder')
parser.add_argument('--with_thermal', default=False, action="store_true", help='use thermal images')
parser.add_argument('--metric', default=False, action="store_true", help='triplet loss enable')
parser.add_argument('--membership', default=False, action="store_true", help='membership loss enable')
parser.add_argument('--custom_sampler', default=False, action="store_true", help='use custom sampler')
parser.add_argument('--num_instances', type=int, default=2, metavar='N', help='# of minimum instances')
parser.add_argument('--transfer', default=False, action="store_true", help='transfer method enable')
parser.add_argument('--soft_label', default=False, action="store_true", help='soft label enable')
parser.add_argument('--not_test_ODIN', default=True, action="store_false", help='if do not want test ODIN, check this option')
parser.add_argument('--train', default=False, action="store_true", help='Train models')
parser.add_argument('--test', default=False, action="store_true", help='Test models')
parser.add_argument('--model_path', type=str, help='**test mode only** trained model path')


-d mobticon --train --num_classes 5 --not_test_ODIN --soft_label --custom_sampler --num_instances 2 --gpu 2 -e 30 -b 16 -m resnet50 --where server2 -l 5e-3 --with_thermal


--dataset 데이터셋 선택(현재는 mobticon만 가능)
--train training or testing 선택
--same_class 같은 class로 간주 (ex- 4, 5번 class를 하나의 class로 보고싶다면 --same_class 4 5)
--except_class class에서 제외
--OOD_class 사용 x
--num_classes 5 class이지만 --same_class 또는 --except_class를 사용한다면 이에 맞춰서 class 개수 지정
--net_type(-m) resnet계열 사용해주시면 됩니다
--board_clear tensorboard 모두 제거
--with_thermal thermal image와 visual image를 concatenate하여 사용
--custom_sampler data imbalance를 해결하기 위해 사용
--num_instances batch에 한 class의 instance가 최소 --num_instances만큼 포함(코딩 실수로 batchsize/8로 설정)
--soft_label label smoothing을 위한 parameter
--model_path testing시 model_path 꼭 명시해줄 것 ./model/checkpoint_last.pth.tar 등....
