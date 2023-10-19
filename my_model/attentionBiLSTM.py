import torch
import torch.nn as nn
import torch.nn.functional as F
from models_.attention import AttentionLayer
from models_.autoencoder import LeftArmAutoencoder, RightArmAutoencoder, TrunkAutoencoder, LeftLegAutoencoder, RightLegAutoencoder
from models_.bilstm import BiLSTM

# Define the indices for each body part
# print(keypoints_sequences)
left_arm_indices = [5, 7, 9]
right_arm_indices = [6, 8, 10]
trunk_indices = [0, 1, 2, 3, 4, 5, 6, 11, 12]
left_leg_indices = [12, 14, 16]
right_leg_indices = [11, 13, 15]

class ActionRecognizationModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, rightArm_input, leftArm_input, trunk_input, rightLeg_input, leftLeg_input):
        super(ActionRecognizationModel, self).__init__()
        self.attention = AttentionLayer(hidden_dim)
        self.bilstm = BiLSTM(input_dim, hidden_dim, num_layers, num_classes, sequence_length=75)
        self.rightArmAutoencoder = RightArmAutoencoder(rightArm_input)
        self.leftArmAutoencoder = LeftArmAutoencoder(leftArm_input)
        self.truckAutoencoder = TrunkAutoencoder(trunk_input)
        self.rightLegAutoencoder = RightLegAutoencoder(rightLeg_input)
        self.leftLegAutoencoder = RightLegAutoencoder(leftLeg_input)
        self.fc = nn.Linear(150 * 66, num_classes)  # Linear classification layer        

    def forward(self, keypoints_sequences, bbox_sequences):  #input1 - right arm, input2 - left arm, input3 - trunk, input4 - right leg, input5 - left leg

        #feed into 5 autoencoders
        sequences_combined_outputs = self.autoencoders_extract(keypoints_sequences, bbox_sequences)    

        #feed into attention mechasnism
        attention_output = self.attention_combine(sequences_combined_outputs)

        # Convert the list of tensors into a single tensor
        lstm_input = torch.stack(attention_output, dim=0)
        print("feed into LSTM shape: ", lstm_input.shape)

        lstm_input = lstm_input.view(150, 1, 1)

        lstm_output = self.bilstm(lstm_input)
        print("LSTM output: ", lstm_output.shape)

        # # Classification layer
        lstm_output = lstm_output.view(-1)
        action_scores = self.fc(lstm_output)

        # Apply sigmoid activation to get probabilities
        action_probs = torch.sigmoid(action_scores)

        return action_probs
    
    def autoencoders_extract(self, keypoints_sequences, bbox_sequences):
        print("keypoints: ", len(keypoints_sequences), " bbox", len(bbox_sequences))

        output_sequences = []


        # for i in range(len(keypoints_sequences)):
        #     keypoints_sequences[i] = keypoints_sequences[i].squeeze()
        #     print("---------------->>",keypoints_sequences)

        for keypoints, bbox_info in zip(keypoints_sequences, bbox_sequences):
                with open("my_list.txt", "w") as file:
                    file.write(str(keypoints))

                keypoints = keypoints.squeeze()

                left_arm_keypoints = keypoints[left_arm_indices] 
                right_arm_keypoints = keypoints[right_arm_indices]
                trunk_keypoints = keypoints[trunk_indices]
                left_leg_keypoints = keypoints[left_leg_indices]
                right_leg_keypoints = keypoints[right_leg_indices]

                print("1----------",left_arm_keypoints)

                left_arm_keypoints = left_arm_keypoints.view(-1)
                right_arm_keypoints = right_arm_keypoints.view(-1)
                trunk_keypoints = trunk_keypoints.view(-1)
                left_leg_keypoints = left_leg_keypoints.view(-1)
                right_leg_keypoints = right_leg_keypoints.view(-1)
                bbox_info = bbox_info.view(-1)

                left_arm_keypoints = left_arm_keypoints.float()
                right_arm_keypoints = right_arm_keypoints.float()
                trunk_keypoints = trunk_keypoints.float()
                left_leg_keypoints = left_leg_keypoints.float()
                right_leg_keypoints = right_leg_keypoints.float()

                rightArm_output = self.rightArmAutoencoder(left_arm_keypoints)
                leftArm_output = self.leftArmAutoencoder(right_arm_keypoints)
                trunk_output = self.truckAutoencoder(trunk_keypoints)
                rightLeg_output = self.rightLegAutoencoder(left_leg_keypoints)
                leftLeg_output = self.leftLegAutoencoder(right_leg_keypoints)
                
                # Check if my_variable is a PyTorch tensor
                trunk_output = torch.mean(trunk_output, dim=0, keepdim=True) #make the trunk into 1 dim
                
                #combine output from 5 autoencoder
                combined_outputs = torch.cat([rightArm_output, leftArm_output, trunk_output, rightLeg_output, leftLeg_output, bbox_info], dim=0)
                # Check if my_variable is a PyTorch tensor

                output_sequences.append(combined_outputs)                

        return output_sequences
    
    def attention_combine(self, inputs):
        output_sequences = []
        for input in inputs:
            # with open("input.txt", "a") as file:  # Use "a" for append mode
            #     file.write(str(input) + '\n')
            print("shape---",input.shape)
            
            if input.shape[0] < 33:
                # Calculate the padding length
                padding_length = 33 - input.shape[0]
                # Create a tensor with zeros for padding
                padding = torch.zeros(padding_length, dtype=input.dtype, device=input.device)
                # Concatenate the input tensor with the padding
                input = torch.cat((input, padding), dim=0)

            # print(input.shape)
            output = self.attention(input)
            output_sequences.append(output)
        
        with open("attention_output.txt", "w") as file:
            file.write(str(output_sequences))
        return output_sequences



