""" Onmt NMT Model base class definition """
import torch.nn as nn


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (:obj:`EncoderBase`): an encoder object
      decoder (:obj:`RNNDecoderBase`): a decoder object
      multi<gpu (bool): setup for multigpu support
    """

    def __init__(self, encoder1, encoder2, encoder3, decoder, multigpu=False):
        self.multigpu = multigpu
        super(NMTModel, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3

        self.decoder = decoder

    def forward(self, src1, src2, src3, tgt,
                lengths1, lengths2, lengths3, dec_state=None):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (:obj:`Tensor`):
                a source sequence passed to encoder.
                typically for inputs this will be a padded :obj:`LongTensor`
                of size `[len x batch x features]`. however, may be an
                image or other generic input depending on encoder.
            tgt (:obj:`LongTensor`):
                 a target sequence of size `[tgt_len x batch]`.
            lengths(:obj:`LongTensor`): the src lengths, pre-padding `[batch]`.
            dec_state (:obj:`DecoderState`, optional): initial decoder state
        Returns:
            (:obj:`FloatTensor`, `dict`, :obj:`onmt.Models.DecoderState`):

                 * decoder output `[tgt_len x batch x hidden]`
                 * dictionary attention dists of `[tgt_len x batch x src_len]`
                 * final decoder state
        """
        tgt = tgt[:-1]  # exclude last target from inputs

        enc_final1, memory_bank1 = self.encoder(src1, lengths1)
        enc_final2, memory_bank2 = self.encoder(src2, lengths2)
        enc_final3, memory_bank3 = self.encoder(src3, lengths3)

        if isinstance(enc_final1, tuple):
            enc_final = ((enc_final1[0] + enc_final2[0] + enc_final3[0]) / 3.,
                         (enc_final1[1] + enc_final2[1] + enc_final3[1]) / 3.)
        else:
            enc_final = (enc_final1 + enc_final2 + enc_final3) / 3.

        enc_state = \
            self.decoder.init_decoder_state(src1, memory_bank1, enc_final)

        decoder_outputs, dec_state, attns = \
            self.decoder(tgt, memory_bank1, memory_bank2, memory_bank3,
                         enc_state if dec_state is None
                         else dec_state,
                         memory_lengths1=lengths1,
                         memory_lengths2=lengths2,
                         memory_lengths3=lengths3)
        if self.multigpu:
            # Not yet supported on multi-gpu
            dec_state = None
            attns = None
        return decoder_outputs, attns, dec_state
