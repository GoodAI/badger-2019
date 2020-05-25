.. architecture of badger implementation


Architecture
============

The Badger architecture (Rosa, et.al 2019) consists of an agent containing many experts. Each expert operates on the same fixed policy but has unique internal states.

The class :py:class:: MultitaskAgent defines the architecture of the agent. It consists of the :py:class:: BadgerAgent and :py:class:: AttentionLayer classes.

The class :py:class:: BadgerAgent consists of key, value and query networks.

