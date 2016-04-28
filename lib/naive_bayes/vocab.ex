defmodule Vocab do
  defstruct tokens: %{}

  def seen_token(vocab, token) do
    put_in(vocab.tokens[token], true)
  end

  def remove_token(vocab, token) do
    Map.delete(vocab, token)
  end
end
