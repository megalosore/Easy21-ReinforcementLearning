# Easy21 Reinforcement Learning - COMPM050

This repository contains my solutions to the **Easy21 assignment** from UCL’s **COMPM050 Advanced Topics in Reinforcement Learning** course. The goal of the assignment is to apply practical reinforcement learning methods to a simplified Blackjack-like card game called **Easy21**, exploring both tabular and function approximation approaches.

---

## Project Overview

**Easy21** is a card game with the following rules:

- Each draw results in a card value 1–10 with a color (red probability 1/3, black probability 2/3).  
- Player starts with one black card; dealer also starts with one black card.  
- Player may **hit** or **stick**; going above 21 or below 1 is a bust (reward -1).  
- Dealer always sticks at 17 or above, hits otherwise.  
- Rewards: win (+1), lose (-1), draw (0).  

The assignment focuses on **model-free RL**, without explicitly representing the MDP transitions.

---

## Implemented Methods

1. **Monte-Carlo Control**  
   - Tabular value function initialized to zero.  
   - Time-varying step-size: αₜ = 1 / N(sₜ, aₜ)  
   - ϵ-greedy exploration with ϵₜ = N₀ / (N₀ + N(sₜ))  
   - Computes the optimal value function V\*(s) = max_a Q\*(s,a).  

2. **Sarsa(λ) (TD Learning)**  
   - Tabular implementation with eligibility traces.  
   - Experiments over λ ∈ {0, 0.1, …, 1}.  
   - Tracks convergence via mean-squared error against the Monte-Carlo solution.  

3. **Linear Function Approximation**  
   - Coarse-coded binary features over a 3×6×2 cuboid for (dealer, player, action).  
   - Linear Q-function: Q(s,a) = φ(s,a)ᵀ θ  
   - Sarsa(λ) with constant step-size and exploration.  

---

## References

- **COMPM050 / COMPGI13: Advanced Topics in Reinforcement Learning** - UCL course page  
  [http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

- **Silver, D. (2015). Lectures on Reinforcement Learning** - Online video lectures by David Silver  
  [https://www.davidsilver.uk/teaching/](https://www.davidsilver.uk/teaching/)

- **DeepMind 2015 Reinforcement Learning YouTube Playlist** - Practical RL lectures  
  [https://www.youtube.com/playlist?list=PLvaXXemMsV5DYxQxnQ2PRmVOyXoLHZSa0](https://www.youtube.com/playlist?list=PLvaXXemMsV5DYxQxnQ2PRmVOyXoLHZSa0)

- **Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd Edition). MIT Press** - Foundational textbook in reinforcement learning
