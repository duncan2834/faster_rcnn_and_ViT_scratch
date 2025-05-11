import torch
import torch.nn as nn
import torchvision
import math
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_iou(boxes1, boxes2): # Tính IOU 
    # boxes1 : Nx4, boxes2: Mx4, IOU shape: NxM
    # area = (x2 - x1) * (y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1]) 
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1]) 
    
    # lấy x1, y1 (topleft)
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])
    
    # lấy x2, y2 (bottomright)
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area
    return intersection_area / union # (N, M) cong thuc tinh iou
    
def apply_rgs_head_to_anchors_or_proposals(box_pred, anchors_or_proposals): # chuyen anchorbox sang proposal, hoac la tu proposal sang detect
    # box_pred : (num_anchors_or_proposals, num_classes, 4)
    # anchor_or_proposals: (num_anchors_or_proposals, 4)
    # return pred_boxes : (num_anchors_or_proposals, num_classes, 4)
    
    box_pred = box_pred.reshape(box_pred.size(0), -1, 4)
    
    # lay cx, cy, w, h tu x1, y1, x2, y2
    
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h
    
    dx = box_pred[..., 0]
    dy = box_pred[..., 1]
    dw = box_pred[..., 2]
    dh = box_pred[..., 3] # (num_anchors_or_proposals, num_classes)
    
    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=math.log(1000.0 / 16))
    dh = torch.clamp(dh, max=math.log(1000.0 / 16))

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None] # (num_anchors_or_proposals, num_classes)
    pred_h = torch.exp(dh) * h[:, None]
    
    pred_box_x1 = pred_center_x - pred_w/2
    pred_box_y1 = pred_center_y - pred_h/2
    pred_box_x2 = pred_center_x + pred_w/2
    pred_box_y2 = pred_center_y + pred_h/2
    
    pred_boxes = torch.stack((pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2) # (num_anchors_or_proposals, num_classes, 4)
    return pred_boxes


def clamp_boxes_to_image(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    
    h, w = image_shape[-2:]
    
    boxes_x1 = boxes_x1.clamp(min=0, max=w)
    boxes_y1 = boxes_y1.clamp(min=0, max=h)
    boxes_x2 = boxes_x2.clamp(min=0, max=w)
    boxes_y2 = boxes_y2.clamp(min=0, max=h)
    
    boxes = torch.cat((
        boxes_x1[..., None],
        boxes_y1[..., None],
        boxes_x2[..., None],
        boxes_y2[..., None]
    ), dim=-1)
    return boxes

def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    # lấy cx, cy, h, w từ x1,y1,x2,y2 cho anchors
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + widths*0.5
    center_y = anchors_or_proposals[:, 1] + heights*0.5
    
    # lấy cx, cy, h, w từ x1,y1,x2,y1 cho gt boxes
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + gt_widths*0.5
    gt_center_y = ground_truth_boxes[:, 1] + gt_heights*0.5
    
    target_dx = (gt_center_x - center_x) / widths
    target_dy = (gt_center_y - center_y) / heights
    target_dw = torch.log(gt_widths/widths)
    target_dh = torch.log(gt_heights/heights)
    
    regression_targets = torch.stack((
        target_dx,
        target_dy,
        target_dw,
        target_dh
    ), dim=1)
    return regression_targets

def sample_positive_negative(labels, positive_count, total_count): # pos count thuong la 128, total = 256
    positive = torch.where(labels >= 1)[0] # lay tat ca cac label pos
    negative = torch.where(labels == 0)[0] # lay tat ca cac label neg
    num_pos = positive_count # 128
    num_pos = min(positive.numel(), num_pos) # nghia la chi lay max la 128 thoi, tat ca cac label pos co the < or > 128
    num_neg = total_count - positive_count # 128
    num_neg = min(negative.numel(), num_neg) 
    
    perm_positive_idxs = torch.randperm(positive.numel(), 
                                        device=positive.device)[:num_pos] # lay bua num_pos cai pos
    perm_negative_idxs = torch.randperm(negative.numel(),
                                        device=negative.device)[:num_neg] # lay bua num_neg cai neg
    pos_idxs = positive[perm_positive_idxs] # idx cua cac label pos da lay bua
    neg_idxs = negative[perm_negative_idxs] # idx cua cac label neg da lay bua
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask

def transform_boxes_to_original_size(boxes, new_size, original_size):
    # chuyen box ve size cu
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class RPN_scratch(nn.Module):
    def __init__(self, in_channels=512):
        super().__init__()
        self.scale = [128, 256, 512] # 128x128, 256x256, 512x512
        self.ratio = [0.5, 1, 2] # 1:2, 1:1, 2:1
        self.k = len(self.scale) * len(self.ratio) # 3 * 3 = 9, số anchors
        self.rpn_topk = 2000 if self.training else 300
        self.rpn_prenms_topk = 12000 if self.training else 6000
        self.relu = nn.ReLU()
        self.rpn = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1) # 3x3
        self.cls = nn.Conv2d(in_channels=in_channels, out_channels=self.k, kernel_size=1, stride=1) # 1x1
        self.rgs = nn.Conv2d(in_channels=in_channels, out_channels=4*self.k, kernel_size=1, stride=1) # 1x1
        
        for layer in [self.rpn, self.cls, self.rgs]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)
            
    def filter_proposals(self, proposals, cls_score, image_shape):
        # Pre NMS filtering
        cls_score = cls_score.reshape(-1)
        cls_score = torch.sigmoid(cls_score)
        
        _, top_n_idx = cls_score.topk(min(self.rpn_prenms_topk, len(cls_score)))
        
        cls_score = cls_score[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Khớp boxes vào các biên ảnh
        proposals = clamp_boxes_to_image(proposals, image_shape)
        
        # Small boxes based on width and height filtering
        min_size = 16
        ws, hs = proposals[:, 2] - proposals[:, 0], proposals[:, 3] - proposals[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        proposals = proposals[keep]
        cls_score = cls_score[keep]
        
        # NMS based on objectness
        keep_mask = torch.zeros_like(cls_score, dtype=torch.bool)
        keep_indices = torch.ops.torchvision.nms(proposals, cls_score, 0.7)
        keep_mask[keep_indices] = True
        keep_indices = torch.where(keep_mask)[0]
        
        post_nms_keep_indices = keep_indices[
            cls_score[keep_indices].sort(descending=True)[1]
        ]
        
        # Post NMS topk filtering
        proposals, cls_score = (proposals[post_nms_keep_indices[:self.rpn_topk]],
                                cls_score[post_nms_keep_indices[:self.rpn_topk]])
        
        return proposals, cls_score
        
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        # lấy IOU (gt_boxes, num_anchors)
        iou_matrix = get_iou(gt_boxes, anchors)
        
        # Với mỗi anchor, lấy cái gt box index tốt nhất
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        
        # copy để add low quality box
        best_match_gt_idx_pre_thresholding = best_match_gt_idx.clone()
        
        below_low_threshold = best_match_iou < 0.3
        between_threshold = (best_match_iou >= 0.3) & (best_match_iou < 0.7)
        
        best_match_gt_idx[below_low_threshold] = -1 # cái này là neg, dùng để phân loại được
        best_match_gt_idx[between_threshold] = -2 # ko dùng cái này, giữa giữa ko dùng để lgi
        
        # low quality anchor box, là mấy cái anchor box có độ overlap cao nhất với một cái gt box đã cho
        # với mỗi gt box, lấy cái IOU to nhất trong tất cả các anchors
        best_anchor_iou_for_gt, _ = iou_matrix.max(dim=1) # (số gt box trong ảnh)
        
        # lấy các anchor có IOU = IOU highest với mỗi gt box
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchor_iou_for_gt[:, None])
        
        # lấy các anchor index để update
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        
        best_match_gt_idx[pred_inds_to_update] = best_match_gt_idx_pre_thresholding[pred_inds_to_update]
        
        matched_gt_boxes = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        # Set all foreground anchor labels as 1
        labels = best_match_gt_idx >= 0
        labels = labels.to(dtype=torch.float32)
        
        # Set all background anchor labels as 0
        background_anchors = best_match_gt_idx == -1
        labels[background_anchors] = 0.0
        
        # Set all to be ignored anchor labels as -1
        ignored_anchors = best_match_gt_idx == -2
        labels[ignored_anchors] = -1.0
        
        return labels, matched_gt_boxes
    
    def generate_anchors(self, image, feat): # feat: feature map
        grid_h, grid_w = feat.shape[-2:] # lay h voi w 
        image_h, image_w = image.shape[-2:] 
        
        stride_h = torch.tensor(image_h // grid_h,
                                dtype=torch.int64,
                                device=feat.device)
        
        stride_w = torch.tensor(image_w // grid_w,
                                dtype=torch.int64,
                                device=feat.device)

        scales = torch.as_tensor(self.scale,
                                dtype=feat.dtype,
                                device=feat.device)
        ratios = torch.as_tensor(self.ratio,
                                dtype=feat.dtype,
                                device=feat.device)   
        # Lấy h*w = 1 và có h/w = ratios
        h_ratio = torch.sqrt(ratios)
        w_ratio = 1 / h_ratio
        
        # có h, w ratio rồi thì chỉ cần nhân với scale là ra được các anchor box với scale khác nhau
        
        ws = (w_ratio[:, None]) * scales[None, :].view(-1)
        hs = (h_ratio[:, None]) * scales[None, :].view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=1) / 2
        base_anchors = base_anchors.round()
        
        # Shifts hết chiều rộng theo từng vị trí  [0, 1, .... w_feat - 1] * stride_w
        shifts_x = torch.arange(0, grid_w,
                                dtype=torch.int32,
                                device=feat.device) * stride_w
        shifts_y = torch.arange(0, grid_h,
                                dtype=torch.int32,
                                device=feat.device) * stride_h
        
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # (h_feat, w_feat)
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        
        shifts = torch.stack((shifts_x, shifts_y, shifts_x, shifts_y), dim=1) # tọa độ tâm của tất cả vị trí trên feature map (H_feat * W_feat, 4) x1 y1 x2 y2 tuong ung voi -ws, -hs , ws , h2  / 2 của base anchor
        # base_anchor -> (num_anchor_per_location, 4)
        
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)) # cộng vô để ra các anchor box của tất cả vị trí trên featmap
        # (H_feat * W_feat, num_anchors_per_location=k, 4)
        anchors = anchors.reshape(-1, 4) # (H_feat * W_feat * num_anchors_per_location, 4)
        
        return anchors # tất cả anchor box tại tất cả các vị trí
    
    def forward(self, image, feat, target):
        # cho qua rpn -> cls + rgs
        rpn_feat = self.relu((self.rpn(feat)))
        cls_score = self.cls(rpn_feat)
        box_pred = self.rgs(rpn_feat)
        
        # generate anchors
        anchors = self.generate_anchors(image, feat)
        
        # cls_score -> batch, num_anchors_per_location, H_feat, W_feat
        num_anchors_per_location = cls_score.size(1)
        cls_score = cls_score.permute(0, 2, 3, 1) # -> batch, H, W, num
        
        cls_score = cls_score.reshape(-1, 1) # -> batch * h * w * num, 1
        
        box_pred = box_pred.view(
            box_pred.size(0),
            num_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-1]
        )
        box_pred = box_pred.permute(0, 3, 4, 1, 2)
        box_pred = box_pred.reshape(-1, 4)
        
        proposals = apply_rgs_head_to_anchors_or_proposals(box_pred.detach().reshape(-1, 1, 4), anchors)
        proposals = proposals.reshape(proposals.size(0), 4) # các box tiềm năng
        
        proposals, scores = self.filter_proposals(proposals, cls_score.detach(), image.shape)
        rpn_output = {
            'proposals': proposals,
            'scores': scores
        }
        if not self.training or target is None:
            # không train thì không cần làm j
            return rpn_output
        else:
            # assign gt box và label cho mỗi anchor
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0])
            
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_anchors, anchors)
            
            # Sample positive, negative anchors để train
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels_for_anchors, positive_count=128, total_count=256)
            sampled_idx = torch.where(sampled_neg_idx_mask | sampled_pos_idx_mask)[0]
            localization_loss = (
                torch.nn.functional.smooth_l1_loss(
                    box_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta=1/9,
                    reduction="sum",
                ) / (sampled_idx.numel())
            )
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_score[sampled_idx].flatten(),
                labels_for_anchors[sampled_idx].flatten()
            )
            
            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output
            

class ROIHead(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        super().__init__()
        self.fc_inner_dim = 1024
        self.pool_size = 7
        self.num_classes = num_classes
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
    
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        # lấy iou matrix giữa proposals và gt box
        iou_matrix = get_iou(gt_boxes, proposals)
        
        # với mỗi proposal, tìm gt box có iou lớn nhất
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        background_proposals = (best_match_iou < 0.5) & (best_match_iou >= 0.1)
        ignored_proposals = best_match_iou < 0.1
        
        best_match_gt_idx[background_proposals] = -1
        best_match_gt_idx[ignored_proposals] = -2
        
        # lấy các GTBOX nào fit nhất với TẤT CẢ proposals, nghĩa là proposal trông n cực kỳ ko fit với cái nào thì vẫn lấy cái fit nhất mặc dù nó bé
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)] # clamp min = 0 là giá trị nào < 0 thì -> 0
        
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        labels[background_proposals] = 0 # background
        
        labels[ignored_proposals] = -1 # ignored
        return labels, matched_gt_boxes_for_proposals
    
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        # bỏ các box điểm thấp
        keep = torch.where(pred_scores > 0.05)[0] # lay index
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]

        # bỏ các box nhỏ
        min_size = 1
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1] # x2 - x1, y2 - y1
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]
        
        # Class wise nms
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(class_id == pred_labels)[0]
            curr_keep_indices = torch.ops.torchvision.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                0.5
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(descending=True)[1]]
        
        keep = post_nms_keep_indices[:100] # giu lai 100 thang score cao nhat
        
        pred_boxes, pred_labels, pred_scores = pred_boxes[keep], pred_labels[keep], pred_scores[keep]
        return pred_boxes, pred_labels, pred_scores
                    
    def forward(self, feat, proposals, image_shape, target):
        if self.training and target is not None:
            # Add gt box cho cai proposal
            proposals = torch.cat([proposals, target['bboxes'][0]], dim=0)
            
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
            
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(proposals, gt_boxes, gt_labels)
            
            # dùng lại hàm sample pos,neg của rpn
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(labels, positive_count=32, total_count=128) 
            
            sampled_idxs = torch.where(sampled_neg_idx_mask | sampled_pos_idx_mask)[0]
            
            # chỉ giữ các sampled proposals
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            
            regression_targets = boxes_to_transformation_targets(matched_gt_boxes_for_proposals, proposals)
            # regression_targets -> (sampled_training_proposals, 4)
            # matched_gt_boxes_for_proposals -> (sampled_training_proposals, 4) 
        
        # VGG16 lam backbone, nen scale la 1/16 (lop maxpool cuoi thay bang cai ROI pooling)
        spatial_scale = 1/16
        # ROI pooling và gọi các layer để predict
        proposal_roi_pool_feats = torchvision.ops.roi_pool(feat, [proposals], output_size=self.pool_size, spatial_scale = spatial_scale)
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6))
        
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
        # cls_scores -> (proposals, num_classes)
        # box_transform_pred -> (proposals, num_classes * 4)
        
        num_boxes, num_classes = cls_scores.shape 
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)
        frcnn_output = {}
        
        if self.training and target is not None:
            classification_loss = torch.nn.functional.cross_entropy(
                cls_scores,
                labels
            )
            # Tính localization loss cho các non-background proposals
            fg_proposals_idxs = torch.where(labels > 0)[0]
            # lấy class label cho các pos proposals trên
            fg_cls_labels = labels[fg_proposals_idxs]
            
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[fg_proposals_idxs, fg_cls_labels],
                regression_targets[fg_proposals_idxs],
                beta=1/9,
                reduction='sum'
            )
            localization_loss = localization_loss / labels.numel()
            frcnn_output['frcnn_classification_loss'] = classification_loss
            frcnn_output['frcnn_localization_loss'] = localization_loss
            
        if self.training:
            return frcnn_output
        else:
            device = cls_scores.device
            # chuyển proposals sang prediction, dùng cái hàm chuyển anchors -> proposals luôn
            pred_boxes = apply_rgs_head_to_anchors_or_proposals(box_transform_pred, proposals)
            pred_scores = torch.nn.functional.softmax(cls_scores, dim=-1)
            
            # clamp box đến biên của ảnh
            pred_boxes = clamp_boxes_to_image(pred_boxes, image_shape)
            
            # tạo label cho các prediction
            pred_labels = torch.arange(num_classes, device=device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
            
            # xóa các prediction là background
            pred_labels = pred_labels[:, 1:]
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            
            # pred_boxes -> (number_proposals, num_classes - 1, 4)
            # pred_scores -> (number_proposals, num_classes - 1)
            # pred_labels -> (number_proposals, num_classes - 1)
            
            # batch everything, by making every class prediction be a separate instance
            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)
            
            pred_boxes, pred_labels, pred_scores = self.filter_predictions(pred_boxes, pred_labels, pred_scores)
            frcnn_output['boxes'] = pred_boxes
            frcnn_output['scores'] = pred_scores
            frcnn_output['labels'] = pred_labels
            return frcnn_output            

class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        vgg16 = torchvision.models.vgg16(pretrained=True)
        self.backbone = vgg16.features[:-1] # ko lay lop pool cuoi 
        self.rpn = RPN_scratch(in_channels=512)
        self.roi_head = ROIHead(num_classes=num_classes, in_channels=512)
        
        
        for layer in self.backbone[:10]: # Freeze cac layer dau
            for p in layer.parameters():
                p.requires_grad = False
            
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = 600
        self.max_size = 1000
        
    def normalize_resize_image_and_boxes(self, image, bboxes):
        dtype, device = image.dtype, image.device
        # Normalize
        mean = torch.as_tensor(self.image_mean, dtype=dtype, device=device)
        std = torch.as_tensor(self.image_std, dtype=dtype, device=device)
        
        image = (image - mean[:, None, None]) / std[:, None, None]  
        
        # Resize, dim be hon thi la 600, dim lon hon thi max la 1000
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(float(self.min_size) / min_size, float(self.max_size) / max_size) # w/600, h/100, cái nào bé hơn thì lấy
        scale_factor = scale.item()
        
        # Resize image dua tren scale
        image = torch.nn.functional.interpolate(
            image, 
            size=None,
            scale_factor=scale_factor,
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False
        )
        
        if bboxes is not None:
            # Resize boxes 
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=bboxes.device)
                / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            xmin, ymin, xmax, ymax = bboxes.unbind(2)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=2)
            
        return image, bboxes
    def forward(self, image, target=None): # target=None trong inference mode
        old_shape = image.shape[-2:] # h w
        if self.training:
            # normalize, resize boxes
            image, bboxes = self.normalize_resize_image_and_boxes(image, target['bboxes'])
            target['bboxes'] = bboxes
        else:
            image, _  = self.normalize_resize_image_and_boxes(image, None)
        
        # Goi backbone vgg16
        feat = self.backbone(image)
        
        # goi rpn va lay proposals
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        
        # goi roi head va lay detection box
        frcnn_output = self.roi_head(feat, proposals, image.shape[-2:], target) # feat, proposals, imageshape, target
        if not self.training: # inference
            # chuyen box ve image goc
            frcnn_output['boxes'] = transform_boxes_to_original_size(frcnn_output['boxes'], image.shape[-2:] , old_shape)
        return rpn_output, frcnn_output
        