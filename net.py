import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
import logging
import sys
import timm
import torchvision.models as models

def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = True

class BaseNetwork(Enum):
    ALEXNET = "AlexNet"
    MOBILENET = "MobileNet V3"
    EFFICIENTNETv2_S = "EfficientNet v2 S"
    NFNET_F5 = "NFNET F0"
    RESNET_50 = "ResNet 50"

def load_weights(from_, to):
    model_dict = to.state_dict()
    from_dict = from_.state_dict()

    pretrained_dict = {k: v for k, v in from_dict.items() if k in model_dict}

    model_dict.update(pretrained_dict)

    to.load_state_dict(model_dict)

    return to

def get_mobilenet(pretrained):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=pretrained)

    features = model.features

    classifier = model.classifier

    return features, classifier

def get_efficientnetv2_s(pretrained):
    features = timm.create_model("efficientnetv2_s", pretrained=pretrained)

    classifier = nn.Identity()

    features.classifier = nn.Identity()

    return features, classifier

def get_nfnet_f5(pretrained):
    features = timm.create_model("nfnet_f3", pretrained=pretrained)
    classifier = features.head
    features.head = features.head.global_pool

    return features, classifier

def get_resnet_50(pretrained):
    features = models.resnet50(pretrained)
    classifier = features.fc
    features.fc = nn.Identity()
    
    return features, classifier

def get_base_network(name, use_handcrafted_features, img_size, num_classes, pretrained):
    if name == BaseNetwork.ALEXNET.value:
        model_alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=pretrained)
        cnn_out_shape = model_alex.features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=cnn_out_shape, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )

        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            del classifier[6]
        else:
            classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.MOBILENET.value:
        features, classifier = get_mobilenet(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        classifier[0] = nn.Linear(in_features=cnn_out_shape, out_features=1280)

        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            del classifier[-1]
        else:
            classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.EFFICIENTNETv2_S.value:
        features, classifier = get_efficientnetv2_s(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        
        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            classifier = nn.Identity()
        else:
            classifier = nn.Linear(in_features=cnn_out_shape, out_features=num_classes)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.NFNET_F5.value:
        features, classifier = get_nfnet_f5(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        
        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            classifier = nn.Identity()
        else:
            classifier = nn.Linear(in_features=cnn_out_shape, out_features=num_classes, bias=True)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.RESNET_50.value:
        features, classifier = get_resnet_50(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        
        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            classifier = nn.Identity()
        else:
            classifier = nn.Linear(in_features=cnn_out_shape, out_features=num_classes, bias=True)
            logging.info("Not using hand-crafted features for base network.")
    else:
        logging.error(f"Unknown base network name '{name}'.")
        sys.exit(-1)

    return features, classifier


def get_base_networkV3(name, use_handcrafted_features, img_size, num_classes, pretrained, features_num):
    if name == BaseNetwork.ALEXNET.value:
        model_alex = torch.hub.load('pytorch/vision:v0.9.0', 'alexnet', pretrained=pretrained)
        cnn_out_shape = model_alex.features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")

        features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),

            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
            
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=cnn_out_shape, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True)
        )

        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            classifier[1] = nn.Linear(in_features=cnn_out_shape + features_num, out_features=4096, bias=True)
            classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)
            
            logging.info(f"CORRECTION V3: Feed forward first layer will have {classifier[1].in_features} neurons.")
        else:
            classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.MOBILENET.value:
        features, classifier = get_mobilenet(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        classifier[0] = nn.Linear(in_features=cnn_out_shape, out_features=1280)

        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")
            classifier[0] = nn.Linear(in_features=cnn_out_shape + features_num, out_features=1280)
            classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
            
            logging.info(f"CORRECTION V3: Feed forward first layer will have {classifier[0].in_features} neurons.")
        else:
            classifier[-1] = nn.Linear(in_features=1280, out_features=num_classes)
            logging.info("Not using hand-crafted features for base network.")
    elif name == BaseNetwork.RESNET_50.value:
        features, classifier = get_resnet_50(pretrained=pretrained)

        cnn_out_shape = features(torch.zeros(1, 3, img_size, img_size)).view(-1).shape[0]
        logging.info(f"Feed forward first layer will have {cnn_out_shape} neurons.")
        classifier = nn.Linear(in_features=cnn_out_shape, out_features=2048)

        if use_handcrafted_features:
            logging.info("Using hand-crafted features for base network. Removing last fully-connected layer.")

            classifier = nn.Sequential(
                nn.Linear(in_features=cnn_out_shape + features_num, out_features=2048),
                nn.ReLU(),
                nn.Linear(in_features=2048, out_features=num_classes)
            )
            
            logging.info(f"CORRECTION V3: Feed forward first layer will have {classifier[0].in_features} neurons.")
        else:
            logging.info("Not using hand-crafted features for base network.")
    else:
        logging.error(f"Unknown base network name '{name}'.")
        sys.exit(-1)

    return features, classifier


def get_base_network_output_dim(name, img_size):
    if name == BaseNetwork.ALEXNET.value:
        return 4096
    elif name == BaseNetwork.MOBILENET.value:
        return 1280
    elif name == BaseNetwork.EFFICIENTNETv2_S.value:
        return 1280
    elif name == BaseNetwork.NFNET_F5.value:
        return 3072
    elif name == BaseNetwork.RESNET_50.value:
        return 2048
    else:
        logging.error(f"Unknown base network name '{name}'.")
        sys.exit(-1)


class FeatureNetwork1HeadWithHandcrafted(nn.Module):
    def __init__(self, base_network, num_classes, img_size, pretrained, feature_vec_length=68):
        super().__init__()
        self.features_extractor, self.classifier = get_base_network(base_network, use_handcrafted_features=True, img_size=img_size, num_classes=num_classes, pretrained=pretrained)
        self.base_network_output_dim = get_base_network_output_dim(base_network, img_size)
        self.feature_vec_length = feature_vec_length
        
        self.fc_final = nn.Linear(self.feature_vec_length + self.base_network_output_dim, num_classes)

    def forward(self, images, features):
        h = self.features_extractor(images)

        h = torch.flatten(h, 1)

        h = self.classifier(h)

        joint_outputs = torch.cat([features, h], 1)

        return self.fc_final(joint_outputs)


class FeatureNetwork1HeadWithHandcraftedV2(nn.Module):
    def __init__(self, base_network, num_classes, img_size, pretrained, feature_vec_length=68):
        super().__init__()
        self.features_extractor, self.classifier = get_base_network(base_network, use_handcrafted_features=False, img_size=img_size, num_classes=num_classes, pretrained=pretrained)
        self.base_network_output_dim = get_base_network_output_dim(base_network, img_size)
        self.feature_vec_length = feature_vec_length
        
        self.fc_final = nn.Linear(self.feature_vec_length + self.base_network_output_dim, num_classes)
        self.network_in_network = nn.Sequential(
            nn.Linear(self.feature_vec_length, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, images, features):
        h = self.features_extractor(images)

        h = torch.flatten(h, 1)

        network_outs = self.classifier(h)

        feature_outputs = self.network_in_network(features)

        return (network_outs + feature_outputs) / 2

class FeatureNetwork1HeadWithHandcraftedV3(nn.Module):
    def __init__(self, base_network, num_classes, img_size, pretrained, feature_vec_length=68):
        super().__init__()
        self.features_extractor, self.classifier = get_base_networkV3(base_network, use_handcrafted_features=True, img_size=img_size, num_classes=num_classes, pretrained=pretrained, features_num=feature_vec_length)
        self.base_network_output_dim = get_base_network_output_dim(base_network, img_size)
        self.feature_vec_length = feature_vec_length
        
    def forward(self, images, features):
        h = self.features_extractor(images)

        h = torch.flatten(h, 1)
        joint_outputs = torch.cat([features, h], 1)

        network_outs = self.classifier(joint_outputs)
        return network_outs


class FeatureNetwork1HeadNoHandcrafted(nn.Module):
    def __init__(self, base_network, num_classes, img_size, pretrained):
        super().__init__()
        self.features_extractor, self.classifier = get_base_network(base_network, use_handcrafted_features=False, img_size=img_size, num_classes=num_classes, pretrained=pretrained)
        self.base_network_output_dim = get_base_network_output_dim(base_network, img_size)

    def forward(self, images, features):
        h = self.features_extractor(images)

        h = torch.flatten(h, 1)

        return self.classifier(h)