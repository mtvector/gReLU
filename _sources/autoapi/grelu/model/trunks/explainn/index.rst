grelu.model.trunks.explainn
===========================

.. py:module:: grelu.model.trunks.explainn


Classes
-------

.. autoapisummary::

   grelu.model.trunks.explainn.ExplaiNNConvBlock
   grelu.model.trunks.explainn.ExplaiNNTrunk


Module Contents
---------------

.. py:class:: ExplaiNNConvBlock(in_channels: int, out_channels: int, kernel_size: int, groups: int, act_func: str, dropout: float)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional block for the ExplaiNN model.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: ExplaiNNTrunk(in_len: int, channels=300, kernel_size=19)

   Bases: :py:obj:`torch.nn.Module`


   The ExplaiNN model architecture.

   :param n_tasks: number of outputs
   :type n_tasks: int
   :param input_length: length of the input sequences
   :type input_length: int
   :param channels: number of independent CNN units (default=300)
   :type channels: int
   :param kernel_size: size of each unit's conv. filter (default=19)
   :type kernel_size: int


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



