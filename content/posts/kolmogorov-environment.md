+++
title = 'Kolmogorov AI Framework | Part2 - brainfuck'
date = 2025-04-02T14:22:47+01:00
draft = false
+++


![brainfuck meme](/kolmogorov/brainfuck.png)

# Motivations

I recently started a series of article about a framework I was thinking about, which allows agent to be trained in a reinforcement learning basis, executing one action at a time on specific environments.

The first part of the series can be found here: [Kolmogorov AI Framework](https://bornlex.github.io/posts/kolmogorov-ai-framework/).

# Environment

To build this proof of concept, I chose the brainfuck programming language. It is part of the esoteric programming languages family, but it is made of only 6 instructions, making it very easy to start with as an execution environment.

## brainfuck (no capital B please)

### About brainfuck

The brainfuck programming language is a minimalistic imperative programming language, designed by Urban Muller in 1993. The idea was to build a simple language with a compiler as little as possible.

Though Turing-complete, it was not intended for serious use.

As an introduction, let us show what a “Hello world!” program looks like in brainfuck:

```python
++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]>>.>
---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++.
```

Looks like something we want to invest time in right?

After telling more about how brainfuck works, we will get back to this Hello world program and explain what it does, precisely.

### Memory

Before jumping to brainfuck instructions, let us explain a bit how brainfuck handles memory.

brainfuck uses a single array of at least 30.000 cells and a pointer (as stated in the specifications). The pointer starts at 0 (the leftmost cell) and can either be increased (++) or decrease (—). At first, the array is made of 0s.

That’s roughly it.

### Instructions

The brainfuck programming language is only made of 8 instructions:

| Instruction | Meaning |
| --- | --- |
| > | Increases pointer (make it jump one cell to the right) |
| < | Decreases pointer (make it jump one cell to the left) |
| + | Increases the value of the cell the pointer points to |
| - | Decreases the value of the cell the pointer points to |
| . | Outputs ASCII code under pointer |
| , | Reads char and stores its ASCII code under pointer |
| [ | Jumps past the matching ] if the value under the pointer is 0, goes to next instruction otherwise |
| ] | Jumps back to the matching [ if the cell under the pointer is nonzero, goes to next instruction otherwise |

In our first implementation, we will not use the **[** and **]** instructions. Looping makes it harder to follow the flow of execution for a linear model, and while it is handful, we do not need to to build our agent.

In fact, any loop can be written as a sufficiently large number of instructions and if statements, considering that you can write an indefinitely large amount of instructions.

On top of those instructions, I’ve added a **NOP** instruction that is used for the agent to indicate the end of the program.

## Reward

The reward mechanism is very important in a reinforcement learning setup, because it is what gets optimized by the agent. It acts as a loss function (kind of).

In our case, the agent tells the brainfuck interpreter what to do, so it outputs instructions one by one until it either decides to stop by itself (outputing the NOP instruction) or if it reaches the maximum number of steps (gets killed by the environment).

This might need to be tweaked in the future, but the current reward mechanism works as follow:

- every instruction costs a little something, so a negative reward is used for every instruction the program outputs, this is made to make the agent optimize as much as possible
- when the agent decides to stop the execution (NOP) or gets killed by the environment itself (maximum number of steps reached), the final reward is computed using the Levenshtein distance between the expected output and the current state of the standard output.

### Levenshtein distance

Using the Levenshtein distance allows the environment to reward the agent gradually. The closer the agent is to the expected output, the greater the reward.

Here is the code to compute the distance:

```python
def levenshtein_distance(s1, s2):
    # Create a matrix to store distances
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Compute the distances
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1,      # Deletion
                           dp[i][j - 1] + 1,      # Insertion
                           dp[i - 1][j - 1] + cost)  # Substitution

    return dp[m][n]

```

## Gymnasium

Gymnasium is a maintained fork of the Gym library developed by OpenAI for reinforcement learning environments.

It provides the user with a framework to either use an existing environment among a large library or develop his own.

When building a custom environment, we inherit from the **gymnasium.Env** class:

```python
class Brainfuck(gymnasium.Env):
	pass
```

The careful reader will note that I have used a capital B as the first letter. This is indeed very sad to write brainfuck with a capital letter, but I have chosen to follow Python guidelines first.

We then have to implement a few methods for our environment to be usable by agents:

- **constructor**: initialize the action and observation spaces
- **step**: takes an action played by the agent and returns the current state of the environment
- **reset**: restore the environment to its initial parameters, so it can be used for another episode
- **render**: display the environment

### Constructor

The constructor can be anything you want but it needs to store two attributes:

- the **action_space**: defines what actions can be played by the agent
- the **observation_space**: defines what observations look like. An observation is what the agent sees from the environment, it acts based on it.

```python
def __init__(self, memory_size: int = 1000):
	  self.__memory_size = memory_size
    self.__action_size = 5 + 1
    self.__ascii_max_value = 256
    self.__counter_max_value = 1001
    self.__max_signed_value = max(
        self.__ascii_max_value, (self.__counter_max_value - 1) // 2
    )
    self.__standard_output_size = 100
    
	  ## some more attributes init here
	  ## ...
	  ## ...
	  
    self.action_space = gymnasium.spaces.Discrete(self.__action_size)
    self.observation_space = gymnasium.spaces.Dict({
        "pointer": gymnasium.spaces.Discrete(self.__memory_size),
        "memory": gymnasium.spaces.MultiDiscrete([
            max(self.__ascii_max_value, self.__counter_max_value)
            for _ in range(self.__memory_size)
        ]),
        "standard_output": gymnasium.spaces.MultiDiscrete([
            self.__ascii_max_value
            for _ in range(self.__standard_output_size)
        ])
    })
```

The action space is made of only 6 possibilities, the 5 brainfuck instructions and the NOP instruction.

From the agent perspective, an action payed by the agent is an integer between 0 and 5.

The observation space is made of 3 different things:

- the value of the pointer
- the content of the memory
- the content of the standard output

This environment is in complete information, with those 3 values, the agent sees everything there is to be seen.

### step

The step method is used to update the environment based on what the agent chose to do. It updates the environment accordingly and returns the new state of the environment, the latest observation.

```python
def step(
    self, action: ActType
) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    self.__current_step += 1

    if action == Actions.NOP:
        return (
            self.__get_state(),
            reward.levenshtein_distance(
                self.__standard_output.read(),
                self.__expected_output
            ),
            True,
            False,
            {}
        )
    elif action == Actions.INCREMENT_POINTER:
        self.__increment_pointer()
    elif action == Actions.DECREMENT_POINTER:
        self.__decrement_pointer()
    elif action == Actions.INCREMENT_VALUE:
        self.__increment_value()
    elif action == Actions.DECREMENT_VALUE:
        self.__decrement_value()
    elif action == Actions.OUTPUT:
        self.__output()
    else:
        raise NotImplementedError("Action not implemented")

    return (
        self.__get_state(),
        reward.INSTRUCTION_PENALTY,
        False,
        self.__current_step >= self.__max_steps,
        {}
    )
```

The returned value is a tuple containing 5 values:

- the current state of the environment (the observation)
- the reward at this step
- a boolean value that is True if the episode is over, False otherwise
- another boolean value that is True if the episode is truncated, False otherwise
- a dictionary containing meta information, in our case this will not be used, so we return an empty dictionary

The if condition internally executes other methods based on the instruction chosen by the agent.

### reset

The reset method is simpler, it is used only to restore the initial state of the environment so we can use it again, fresh start.

```python
def reset(
    self,
    *,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
):
    super().reset(seed=seed)

    self.__pointer = 0
    self.__current_step = 0
    self.__memory.reset()
    self.__standard_output.reset()

    return (
        self.__get_state(),
        {}
    )
```

### render

The render method is used to display the current state of the environment. It uses a single argument to chose whether we display the environment graphically or not.

```python
def render(self, mode: str = "human") -> Optional[Any]:
    """
    Render the environment.
    """
    if mode == 'human':
        print(self.__standard_output.read())
```

In our case, we will only print the current state of the standard output.

# Code

The code is publicly available here: [Github brainfuck](https://github.com/Bornlex/brainfuck).
