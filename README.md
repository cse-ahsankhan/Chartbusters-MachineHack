# Chartbusters-MachineHack
Minimalistic approach in solving the Chartbusters Hackathon problem (https://www.machinehack.com/course/chartbusters-prediction-foretell-the-popularity-of-songs/).

Note:

I saw alot of possible number of scenarios to solve this problem but was planning on doing this one with a minimalistic possible approach so that it is easy to understand and manipulate.

Technique used:

> Recursive Feature selection <br>
> Datetime manipulation <br>
> Alpha numeric conversions. <br>
> That's it!!!!! <br>

Neural Network design:
Used Keras sequential model inorder to create a three layered neural network (One hidden, One input and One output).

For preprocessing the data we simply use the regular Pandas method and then convert the Pandas dataframe into the Tensorflow dataframe (.h5).


## Result:
<b> This kernel achieved 13th rank in the final tally of the Hackathon(before the end of bounty) </b>
