defmodule Vocab do
  defstruct tokens: %{}

  def seen_token(vocab, token) do
    vocab = put_in(vocab.tokens[token], true)
    vocab
  end
end