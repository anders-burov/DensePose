
# ROUGH DEMO OF HOW ADAPTIVE MARGIN COSINE CAN BE USED.
# DEMO CREATED WHEN I WAS WRITING ADPATIVE MARGIN COSINE WHEN MY IMPLEMENTATION IS STILL FRESH IN MY MIND.
# -------------------------------------------------------------------------------------------------------

# before training begins...

margin_limits = {'max_positive_margin_angle_rad':np.deg2rad(90),
              'min_positive_margin_angle_rad':np.deg2rad(5),
              'max_negative_margin_angle_rad':np.deg2rad(90),
              'min_negative_margin_angle_rad':np.deg2rad(20)}
contrastive_loss = Adaptive_Double_Margin_Contrastive_Loss(option='adaptive double margin cosine', **margin_limits)
distance_type = 'cosine'
print(contrastive_loss.__dict__)

# ....

#for epoch in epochs...
#  for i_batch, data in enumerate(train_dataloader):
    # ...

    all_interx_amts = np.concatenate((interx_amts_1, interx_amts_2))
    # update logbk's intersection amt stats and amt max...
    amt_mean, amt_std = logbk['intersection_amt_stats'].mean, logbk['intersection_amt_stats'].std
    # compute confidences...
    confidences = all_interx_amts / (amt_mean + 2.0 * amt_std)

    input1s, input2s, targets = torch.cat((S_pos1s, S_neg1s)), torch.cat((S_pos2s, S_neg2s)), torch.cat((targets_pos, targets_neg))
    confidences = torch.Tensor(confidences)

    # ...
    net.train()
    optimizer.zero_grad()
    output1s, output2s = net(input1s, input2s)                       
    loss = contrastive_loss(output1s, output2s, targets.float(), confidences)
    loss.backward()
    optimizer.step()