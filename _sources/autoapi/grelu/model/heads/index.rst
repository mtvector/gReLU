grelu.model.heads
=================

.. py:module:: grelu.model.heads

.. autoapi-nested-parse::

   Model head layers to return the final prediction outputs.



Classes
-------

.. autoapisummary::

   grelu.model.heads.ConvHead
   grelu.model.heads.MLPHead


Module Contents
---------------

.. py:class:: ConvHead(n_tasks: int, in_channels: int, act_func: Optional[str] = None, pool_func: Optional[str] = None, norm: bool = False)

   Bases: :py:obj:`torch.nn.Module`


   A 1x1 Conv layer that transforms the the number of channels in the input and then
   optionally pools along the length axis.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the convolutional layer
   :param pool_func: Pooling function.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



.. py:class:: MLPHead(n_tasks: int, in_channels: int, in_len: int, act_func: Optional[str] = None, hidden_size: List[int] = [], norm: bool = False, dropout: float = 0.0)

   Bases: :py:obj:`torch.nn.Module`


   This block implements the multi-layer perceptron (MLP) module.

   :param n_tasks: Number of tasks (output channels)
   :param in_channels: Number of channels in the input
   :param in_len: Length of the input
   :param norm: If True, batch normalization will be included.
   :param act_func: Activation function for the linear layers
   :param hidden_size: A list of dimensions for each hidden layer of the MLP.
   :param dropout: Dropout probability for the linear layers.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      :param x: Input data.



