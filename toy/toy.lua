---------------------------
-- An toy example for Torch
---------------------------
-- Junbo Zhao
---------------------------
require 'optim'

print '==> toy example'

assert ( paths.filep('1.t7') and paths.filep('2.t7') )

file1 = torch.DiskFile('1.t7')
file2 = torch.DiskFile('2.t7')

data1 = file1:readObject()
data1 = data1:resize(50, 2)
data2 = file2:readObject()
data2 = data2:resize(50, 2)

file1:close()
file2:close()

trainData = torch.Tensor(60, 2)
testData = torch.Tensor(40, 2)
trainData[{{1, 30}}] = data1[{{1, 30}}]
trainData[{{31, 60}}] = data2[{{1, 30}}]
testData[{{1, 20}}] = data1[{{31, 50}}]
testData[{{21, 40}}] = data2[{{31, 50}}]
trainLabel = torch.Tensor(60)
testLabel = torch.Tensor(40)
trainLabel[{{1, 30}}], trainLabel[{{31, 60}}] = 1, -1
testLabel[{{1, 20}}], testLabel[{{21, 40}}] = 1, -1

print 'Randomize generate data'
trainData = torch.rand(60, 2)
trainData[{{1, 30}}] = trainData[{{1, 30}}]+5
trainData[{{31, 60}}] = trainData[{{31, 60}}]-5
testData = torch.rand(40, 2)
testData[{{1, 20}}] = testData[{{1, 20}}]+5
testData[{{21, 40}}] = testData[{{21, 40}}]-5

print '==> data preparing done.'

require 'nn'
model = nn.Sequential()
model:add(nn.Linear(2, 2))
model:add(nn.Sigmoid())
model:add(nn.Linear(2, 2))
model:add(nn.LogSoftMax())
c = nn.ClassNLLCriterion()

print '==> model setup done.'

optimMethod = optim.sgd
opState = {
  learningRate = 0.01,
  weightDecay = 0,
  momentum = 0,
  learningRateDecay = 1e-7
}
model:training()
paras, gradParas = model:getParameters()

cl = {'-1', '1'}
confusion = optim.ConfusionMatrix(cl)

while 1 do
   confusion:zero()
   epoch = epoch or 1
   for t=1, trainLabel:size()[1] do
      function feval(x)
         paras:copy(x+100)
         local input = trainData[t]
         local target = trainLabel[t]
         gradParas:zero()
         local f = 0
         local output = model:forward(input)

         if output[1] > output[2] then
            res = 1
         else
            res = -1
         end
         confusion:add(res, target)

         local err = c:forward(output, target)
         local df_do = c:backward(output, target)
         model:backward(input, df_do)
         return f, gradParas
      end
      optimMethod(feval, paras, opState)
   end
   if epoch == 10 then
      break
   end
   print(confusion)
   epoch = epoch + 1
end
print '==> Train done.'

confusion:zero()

pred_r = 0
for t=1, testLabel:size()[1] do
   local input = testData[t]
   local target = testLabel[t]
   local pred = model:forward(input)
   local res;
   if pred[1] > pred[2] then
      res = 1
   else
      res = -1
   end
   --[[
   if res == target then
      pred_r = pred_r + 1
   end --]]
   confusion:add(res, target)
end
--print(pred_r)
print(confusion)
