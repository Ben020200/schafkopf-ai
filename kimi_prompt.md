# PowerPoint Prompt for Kimi

Create a PowerPoint presentation (6 slides) about training a Reinforcement Learning agent to play the German card game Schafkopf. Use a clean, professional design with simple visuals.

## Slide 1: Environment

- **Title:** "Game Environment"
- **Explain:** We built a simulation of Schafkopf using PettingZoo (a multi-agent RL framework)
- **Key points:**
  - 4 players, 32-card Bavarian deck, 8 cards per player
  - Declarer + hidden partner vs 2 opponents (2v2 teams)
  - Goal: Collect 61+ points out of 120 total by winning tricks
  - Environment provides: observations (cards in hand, played cards), legal actions, and rewards
- Include a simple diagram showing 4 players around a table with cards

## Slide 2: Reward Shaping

- **Title:** "Reward Function Design"
- **Problem:** Sparse rewards (only +1/-1 at game end) don't teach the agent which moves are good
- **Solution:** Dense reward shaping - give feedback after every trick
- Show this reward formula:

```
Reward per trick:
• Point differential: (team_gained - opponent_gained) / 120
• High-value capture: +0.02 per ace/ten captured
• Trick win bonus: +0.01
• Threshold bonus: +0.1 when reaching 61 points
```

- Visual: Compare sparse (one signal at end) vs dense (signals throughout game)

## Slide 3: Training (PPO)

- **Title:** "Training with PPO"
- **Algorithm:** Proximal Policy Optimization - stable deep RL method
- **Setup:**
  - 8 parallel game environments
  - PPO agent controls declarer team
  - Rule-based opponent controls enemy team
  - 1 million training steps (~30,000 games)
- Include a simple neural network diagram
- Mention: Actor-Critic architecture with shared layers

## Slide 4: Results - Training Progress

- **Title:** "Training Curve"
- Show a line graph of win rate over training timesteps
- X-axis: Timesteps (0 to 1,000,000)
- Y-axis: Win Rate (0% to 100%)
- Highlight: Win rate starts ~40%, fluctuates, stabilizes around 45-50%
- Note: Agent learns to play better over time through trial and error

## Slide 5: Results - Comparison Table

- **Title:** "Performance Comparison"
- Show a table comparing PPO agent vs baseline agents:

| PPO vs Opponent | Before (sparse) | After (dense) | Improvement |
|-----------------|-----------------|---------------|-------------|
| vs RuleBased    | 37.5%           | 44.6%         | +7.1%       |
| vs Threshold55  | 37.1%           | 46.7%         | +9.6%       |
| vs Random       | 55.2%           | 59.9%         | +4.7%       |

- Key insight: Dense rewards improved performance across all matchups

## Slide 6: Results - Conclusions & Next Steps

- **Title:** "Conclusions & Future Work"
- **What worked:**
  - Dense reward shaping provided better learning signal
  - Training against rule-based opponents (not self-play)
  - PPO algorithm stable and effective
- **Current limitation:** Still loses to rule-based agents (44.6% < 50%)
- **Next steps:**
  - Longer training (5-10 million steps)
  - Larger neural network
  - Curriculum learning (start easy, increase difficulty)
  - Train as both declarer AND opponent teams

## Style Notes

- Use blue/white color scheme
- Keep text minimal, use bullet points
- Add icons for cards, neural networks, and graphs where appropriate
- Use green arrows or highlights for improvements in the results table
