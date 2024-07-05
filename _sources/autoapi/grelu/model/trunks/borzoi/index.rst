grelu.model.trunks.borzoi
=========================

.. py:module:: grelu.model.trunks.borzoi

.. autoapi-nested-parse::

   The Borzoi model architecture and its required classes.



Classes
-------

.. autoapisummary::

   grelu.model.trunks.borzoi.BorzoiConvTower
   grelu.model.trunks.borzoi.BorzoiTrunk


Module Contents
---------------

.. py:class:: BorzoiConvTower(stem_channels: int, stem_kernel_size: int, init_channels: int, out_channels: int, kernel_size: int, n_blocks: int)

   Bases: :py:obj:`torch.nn.Module`


   Convolutional tower for the Borzoi model.

   :param stem_channels: Number of channels in the first (stem) convolutional layer
   :param stem_kernel_size: Width of the convolutional kernel in the first (stem) convolutional layer
   :param init_channels: Number of channels in the first convolutional block after the stem
   :param out_channels: Number of channels in the output
   :param kernel_size: Width of the convolutional kernel
   :param n_blocks: Number of convolutional/pooling blocks, including the stem


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



.. py:class:: BorzoiTrunk(stem_channels: int, stem_kernel_size: int, init_channels: int, n_conv: int, kernel_size: int, channels: int, n_transformers: int, key_len: int, value_len: int, pos_dropout: float, attn_dropout: float, n_heads: int, n_pos_features: int, crop_len: int)

   Bases: :py:obj:`torch.nn.Module`


   Trunk consisting of conv, transformer and U-net layers for the Borzoi model.


   .. py:method:: forward(x: torch.Tensor) -> torch.Tensor

      Forward pass

      :param x: Input tensor of shape (N, C, L)

      :returns: Output tensor



