---
title: "A Bayesian Model for Scoring in Soccer (Part I)"
date: 2022-09-15
categories:
  - blog
tags:
  - bayesian models
  - statistics
  - soccer
  - football
  - canpl
  - xg
---

The Canadian Premier League (CPL), which launched in 2019, is a professional soccer league currently consisting of 8 teams from 5 Canadian provinces. The league has plans to add additional teams in the coming years; being from Vancouver myself, I plan to support the team based in Langley, BC that is planned to debut in 2023. For now though, The league has been incredibly forward-thinking about its analytics, and releases game data for every match that is played. This can be accessed by signing up [here](https://canpl.ca/centre-circle-data/). The data include more than a hundred stats for every team and player in each game.

In this post, I'm going to construct a model for predicting soccer results based on the number of shots, penalties, and [expected goals](https://en.wikipedia.org/wiki/Expected_goals) in previous games. I'm quite new to soccer analytics, so I'm not an expert by any means. That said, the CPL data format differs from other freely available soccer data sources [(like StatsBomb)](https://github.com/statsbomb/open-data) in that it does not include event-level information, so I figured I'd share my exploration!

Expected Goals, or xG, are becoming more and more commonly used in the soccer world. Simply put, xG quantifies how likely an average player is to score on a given shot. For example, a tap-in within the 6 yard box may have an xG of 0.7, meaning an average player would score this chance 70% of the time. A long shot from outside the 18 yard box may have an xG of 0.03, meaning an average player would score this chance just 3% of the time. Of course, the odds of a player scoring a given chance depends on many things, and certain simplifications have to be made to be able to estimate xG reliably. The simplest xG models may only depend on where a shot is taken from, while more complex models may involve a variety of factors like which foot the shot is taken with, the goalkeeper's position, and defender's positions.

So why use xG? Mainly, because goals in soccer are very rare, meaning the final score may not tell the complete story of the game. If two teams play to a 0-0 draw, but team A had a total xG of 3.0 while team B had an xG of 0.3, team A seems more likely to win a future game against team B than vice-versa. There are many short-comings to xG such as differences in striker finishing and goalkeeper shot-stopping abilities. In practice though, it is surprisingly rare for top strikers to consistently out-score their xG.

# The Model

This section gets quite in-depth on the statistics and code that go into the model. If you're more interested in the football side of things, feel free to skip ahead to [the next section](#model-results).

The CPL provides several different data files for each season. Four csv files are provided for the CPL season: CPLPlayerByGame2022.csv, CPLPlayerTotals2022.csv, CPLTeamByGame2022.csv, and CPLTeamTotals2022.csv. In this post, I'll only be using CPLTeamByGame2022.csv. In this file, each game correspods to two rows: one for each team. Each of these rows contains stats such as goals scored, expected goals, and shots.

The overarching goal of this section is to first set up a model for the number of shots taken by a given team in a game against another, then model the distribution of xG values for those shots, and finally to combine these to predict the number of goals, and thus the final result for the game.

## Shots

The first step I'll take is to model the number of shots taken by a given team against a given opponent. Let $\mathcal{T}$ represent the set of $N_\mathrm{teams}$ teams in the league. Let $i,j \in \mathcal{T}$ be the teams facing off in a game. The number of shots for team $i$ against team $j$ is assumed to be drawn from a Poisson distribution:

$$
N^S_{i,j} \sim \mathrm{Poisson}(\lambda^S_{i,j}).
$$

I'll split the shot contribution into penalties and non-penalties (i.e. $N^S_{i,j}=N^P_{i,j}+N^{NP}_{i,j}$), with rate parameters that are given by:

$$
\log{(\lambda^P_{i,j})}=\left\{
\begin{array}{ll}
	\log(\lambda^P)+A_i-D_j+H/2 & i\,\mathrm{is\,at\,home} \\
	\log(\lambda^P)+A_i-D_j-H/2 & j\,\mathrm{is\,at\,home}
\end{array}\right.
$$


$$
\log{(\lambda^{NP}_{i,j})}=\left\{
\begin{array}{ll}
	\log(\lambda^{NP})+A_i-D_j+H/2 & i\,\mathrm{is\,at\,home} \\
	\log(\lambda^{NP})+A_i-D_j-H/2 & j\,\mathrm{is\,at\,home}
\end{array}\right.
$$

Here $\lambda^P$ and $\lambda^{NP}$ are the mean numbers of penalty and non-penalty shots within the league, $A_i$ parameterizes the shot generation of team $i$, $D_j$ parameterizes the shot prevention of team $j$, and $H$ quantifies the home advantage within the league. Given the parameters $A_i^S$, $D_j^S$, and $H$, we would then know the probability of a given number of shots for team $i$ against team $j$. This represents a total of $2N_\mathrm{teams}+1=17$ free parameters.

In order to estimate the values of these parameters, I will use a Bayesian framework. The idea is to use Bayes' Theorem to estimate the values of these parameters, given the observed number of shots in each game that has happened thus far. Writing $\theta$ to represent all of these parameters, we have:

$$
P(\theta\vert\mathrm{Data}) = \frac{P(\mathrm{Data}\vert\theta)P(\theta)}{P(\mathrm{Data})}.
$$

Here, $P(\theta\vert\mathrm{Data})$, referred to as the posterior distribution, gives the probability of a set of parameters $\theta$ given the data. $P(\mathrm{Data}\vert\theta)$, referred to as the likelihood, represents the probability of the data, given a set of parameters $\theta$. $P(\theta)$ is a prior distribution, which represents our knowledge of the parameters $\theta$ before observing the data. The marginal likelihood, $P(\mathrm{Data})$, is the same for all parameter values $\theta$. Thus, if we are only interested in comparing the value of the posterior between different sets of parameters, $\theta$, we can disregard this team. This leaves us needing to evaluate the likelihood and prior in order to estimate the posterior distribtion.

There are many possible options for the prior distributions. I'll adopt some relatively simple and rough estimates for the priors. Ideally, the priors should have only a weak effect on the final parameter estimation. For these parameters, I'll adopt:

$$
A_i \sim \mathrm{Normal}\left(0,0.5\right),
$$

$$
D_i \sim \mathrm{Normal}\left(0,0.5\right),
$$

$$
H \sim \mathrm{Normal}\left(0,0.5\right).
$$

These priors are chosen to be weakly informative and mainly allow the data to speak for itself.

In order to sample from the posterior distribution, I'll use Markov Chain Monte Carlo (MCMC) sampling. I'll use the [PyMC](https://www.pymc.io/welcome.html) python library to implement this. First, we need to load and prepare the data:

```
import pandas as pd
import numpy as np
import os

# Load Centre Circle Data 2022 team data
data_dir='../Centre_Circle_Data/'
team_data=pd.read_csv(os.path.join(data_dir,f'2022 Season',
                                   f'CPLTeamByGame2022.csv'))

# Get Team IDs and Names
team_list=list(team_data['teamId'].unique())
team_names=list(team_data['teamFullName'].unique())

# Convert game data to numpy arrays
i_index=np.array([team_list.index(i_team)
                  for i_team in team_data['teamId']],
                  dtype=int)
j_index=np.array([team_list.index(j_team)
                  for j_team in team_data['opponentId']],
                  dtype=int)
i_home=np.array(team_data['Home'],dtype=int)
shots=np.array(team_data['ShotsTotal'],dtype=int)
penalties=np.array(team_data['PensWon'],dtype=int)
nonpenxg=np.array(team_data['NonPenxG'],dtype=int)
```

With the data loaded into numpy arrays, we can construct the model and sample from the posterior:

```
import pymc as pm

# Calculate mean numbers of shots and penalties
lambdaNP=np.mean(shots)-np.mean(penalties)
lambdaP=np.mean(penalties)

with pm.Model() as model:
    
    # Prior distributions
    A_pm=pm.Normal('A',0,0.5,size=8)
    D_pm=pm.Normal('D',0,0.5,size=8)
    H_pm=pm.Normal('H',0,0.5)
    
    # Non-penalty likelihood distribution
    lambdaNP_ij=lambdaNP*pm.math.exp(
        A[i_index]-D[j_index]-(-1)**i_home*H/2)
    Nnonpenalties_pm=pm.Poisson('Nnonpenalties',
                                lambdaNP_ij,
                                observed=shots-penalties)

    # Penalty likelihood distribution
    lambdaP_ij=lambdaP*pm.math.exp(
        A[i_index]-D[j_index]-(-1)**i_home*H/2)
    Npenalties_pm=pm.Poisson('Npenalties',
                             lambdaP_ij,
                             observed=penalties)

    # Sample from posterior distribution
    samples=pm.sample(2500,chains=4)
```

Once we have samples from the posterior distribution, we can simulate the number of shots and penalties in a game by drawing a sample from the posterior, and then drawing the number of shots and penalties from the resulting Poisson distribution.

## Expected Goals Per Shot

With a model for the number of shots and penalties in a given game, we now need to model the quality of these chances. We can model this using xG. So what does the distribution of xG look like? Unfortunately, we can't reconstruct this directly from the Centre Circle Data set. Since we don't have event-level data, the closest we have is the total xG for each player in each game. For any player with multiple shots in that game, we don't know the individual xG values for each of the shots.

We can get a rough idea of the xG distribution by looking only at players who have a single shot in a given game. The resulting xG distribution looks like this:

![xG distribution](/assets/images/soccer/xgdist.png)

This distribution has three main features:
1. A primary peak around 0.06
2. A secondary peak around 0.25
3. A third peak around 0.79

The third peak is due to penalties. Within this data, each penalty receives an xG of 0.79. The remaining non-penalty xG distribution is bimodal, and looks difficult to model. Why does it have this shape? Ted Knutson's gave a [StatsBomb talk](https://youtu.be/_AYY9XlWEB0), which suggests that this bimodal structure goes away after correcting for the positions of players around the ball when the shot is taken. Keeping this in mind, it's not clear that attempting to model this structure would even be useful if we had the data to do it.

After correcting for nearby player positions, Ted shows that shot xG is distributed more smoothly, with a mode at 0. Inspired by this, I'll model the xG distribution as a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution). With appropriate parameter choice, a beta distribution can resemble the distribution shown by Ted (excluding penalties):

![beta distribution](/assets/images/soccer/beta.png)

The beta distribution is defined via two positive parameters, $\alpha$ and $\beta$. These can be related to the mean ($\mu$) and variance ($\sigma$) via:

$$
\alpha=\mu\left(\frac{\mu(1-\mu)}{\sigma^2}-1\right)
$$

and

$$
\beta=(1-\mu)\left(\frac{\mu(1-\mu)}{\sigma^2}-1\right).
$$

But how can we relate this smooth xG model relate to the bimodal xG distribution observed in the data? The key assumption that I'll use is that while the distribution of xG for individual shots is bimodal, I'll assume that over several shots, the effects of player positioning should average out. In this case, the mean xG for a series of shots should agree on average with the smooth model. Thus, if there are typically about 10 shots per side in a given game, I'll assume that the mean xG per shot from the Centre Circle Data is a reasonable approximation to what one would get for the xG per shot if player positions had been accounted for. Thus, the model should be a reasonable approximation to the observed xG per shot.

With a model for the distribution of xG values for a single shot in hand, we now need a model for the mean xG of $N$ shots. Unfortunately, no nice formula appears to exist for this. Instead, I'll use an approximation. If the distribution of xG for a single shot has mean $\mu$ and variance $\sigma^2$, the mean xG of $N$ shots will have a mean $\mu$ and variance $\sigma^2/N$. It turns out that distribution of the resulting sum is reasonably approximated by a beta distribution, but with this updated mean and variance. In the limit of large $N$, this approximation is exact, since the beta distribution approaches a normal distribution and the central limit theorem can be applied. Even with a relatively small number of shots like 6-10, the approximation turns out to be quite good for parameter values of interest. Here are the distributions of the average of $N$ beta random variables for a few values of $N$:

![average of beta RVs](/assets/images/soccer/beta_av.png)

Finally, we have all the necessary ingredients for our xG model. We will assume that the average non-penalty xG per shot in a game is drawn from a beta distribution:

$$
X_{i,j}\sim\mathrm{Beta}\left(\alpha=\mu^X_{i,j}(N^S_{i,j}/f-1),\beta=(1-\mu^X_{i,j})(N^S_{i,j}/f-1)\right).
$$

Here, $\mu^X_{i,j}$ is an average shot quality for team $i$ against team $j$. $f$ is a global parameter between 0 and 1 that quantifies the variance of the distribution, relative to $\mu^X_{i,j}(1-\mu^X_{i,j})$. $N^S_{i,j}$ is the number of shots taken by team $i$ against team $j$. The form of this equation is chosen such that $\mu^X_{i,j}$ and $f\mu^X_{i,j}(1-\mu^X_{i,j})$ are the mean and variance of xG for individual shots, while $X_{i,j}$ approximates the mean of $N^S_{i,j}$ such shots as being drawn from a beta distribution with the appropriate mean $\mu^X_{i,j}$ and variance $f\mu^X_{i,j}(1-\mu^X_{i,j})/N^S_{i,j}$.

Unlike shots, where each team's attacking or defending ability was added to the logarithm, we need to take care that the mean, $\mu^X_{i,j}$ is between 0 and 1. To do this, instead of using a logarithm, I'll use a function called a [Logit](https://en.wikipedia.org/wiki/Logit) and its inverse, the Expit. The Logit is defined as:

$$
\mathrm{Logit}(x)=\ln\frac{x}{1-x}
$$

and the Expit is:

$$
\mathrm{Expit}(x)=\frac{1}{1+\exp{(-x)}}.
$$

For a value $x$ between 0 and 1, $\mathrm{Logit(x)}$ is between $-\infty$ and $\infty$. Likewise, for $x$ between $-\infty$ and $\infty$, $\mathrm{Expit(x)}$ is between 0 and 1. Both functions are monotonic. To ensure that $\mu^X_{i,j}$ is between 0 and 1, I'll use the following model:

$$
\mathrm{Logit}{(\mu^X_{i,j})}=\left\{
\begin{array}{ll}
	\mathrm{Logit}(\mu^X)+a_i-d_j+h/2 & i\,\mathrm{is\,at\,home} \\
	\mathrm{Logit}(\mu^X)+a_i-d_j-h/2 & j\,\mathrm{is\,at\,home}
\end{array}\right.
$$

Here, $a_i$ indicates how much better team $i$ is at generating high quality changes, $d_i$ indicates how much better team $j$ is at preventing high quality chances, and $h$ represents a home-field advantage in terms of shot quality. I'll use a normal prior of these parameters:

$$
a_i\sim\mathrm{Normal}(0,0.5),
$$

$$
d_i\sim\mathrm{Normal}(0,0.5),
$$

$$
h\sim\mathrm{Normal}(0,0.5),
$$

and a uniform prior on the variance parameter:

$$
f\sim\mathrm{Uniform}(0,1).
$$

The resulting model has $2N_\mathrm{teams}+2$ free parameters. Here's the code for implementing this model in PyMC:

```
with pm.Model() as model:
    
    # Prior distributions
    a_pm=pm.Normal('a',mu=0,sigma=0.5,size=nteams)
    d_pm=pm.Normal('d',mu=0,sigma=0.5,size=nteams)
    h_pm=pm.Normal('h',mu=0,sigma=0.5)
    f_pm=pm.Uniform('f',0.,1.)

    
    # Mask out games with no non-penalty shots and calculate
    # xg per shot
    mask=shots>penalties
    nonpenshots=(shots-penalties)[mask]
    xgpershot=nonpenxg[mask]/nonpenshots
    
    # Likelihood distribution
    mu=pm.math.invlogit(pm.math.logit(np.mean(xgpershot))\
                        +a_pm[i_index[mask]]\
                        -d_pm[j_index[mask]]\
                        -(-1)**i_home[mask]*h_pm/2)
    nonpenxg_pm=pm.Beta(f'nonpenxg',mu*(nonpenshots/f_pm-1),
                        (1-mu)*(nonpenshots/f_pm-1),
                        observed=xgpershot)

    # Sample from posterior distribution
    xg_samples=pm.sample(25000,chains=4)
```

We can simulate the xG for a given shot in a game (given the number of shots) by sampling from this posterior, and then drawing from the relevant beta distribution.

## Expected Goals

Now that we can simulate the shots and expected goals per shot, we have a complete model for expected goals in a given game. Each shot can then be regarded as a Bernoulli trial with the xG value being the probability of scoring. The total score is then the sum of these Bernoulli trials. For non-penalty shots, the xG of the shot is drawn from the Beta distribution model outlined in the previous section. For penalties, the xG is set to 0.79.

The number of non-penalty goals in a game can be thus regarded as being drawn from a Beta-Binomial distribution given the number of shots and the beta distribution parameters governing the xG per shot distribution. This can then be added to the number of penalty goals, determined by drawing from a binomial distribution given the number of penalties and a single trial probability of 0.79. I'll explore the predictive capabilities of this model in the next post. In the remainder of this post, I'll explore the parameter estimates resulting from the samples from the posterior distribution. We can calculate the median value, or credible intervals for each parameter. It will be useful to define the "typical" shots for and against for team $i$ by

$$
\mathrm{Typical\,shots\,for}=\lambda^\mathrm{NP} \exp{(A_i)}
$$

and

$$
\mathrm{Typical\,shots\,against}=\lambda^\mathrm{NP} \exp{(-D_i)},
$$

as well as the typical penalties for and against:

$$
\mathrm{Typical\,penalties\,for}=\lambda^\mathrm{P} \exp{(A_i)}
$$

and

$$
\mathrm{Typical\,penalties\,against}=\lambda^\mathrm{P} \exp{(-D_i)}.
$$

Similarly, we can define:

$$
\mathrm{Typical\,xG\,per\,shot\,for}=\frac{\mu^\mathrm{X}}{\mu^\mathrm{X}+(1-\mu^\mathrm{X})\exp{(-a_i)}}
$$

and

$$
\mathrm{Typical\,xG\,per\,shot\,against}=\frac{\mu^\mathrm{X}}{\mu^\mathrm{X}+(1-\mu^\mathrm{X})\exp{(d_i)}}.
$$

Finally, we'll define the typical goals for and against as:

$$
\begin{array}{l}
\mathrm{Typical\,goals\,for}=\\
(\mathrm{Typical\,shots\,for})(\mathrm{Typical\,xG\,per\,shot\,for}) \\
+0.79(\mathrm{Typical\,penalties\,for})
\end{array}
$$

and

$$
\begin{array}{l}
\mathrm{Typical\,goals\,against}=\\
(\mathrm{Typical\,shots\,against})(\mathrm{Typical\,xG\,per\,shot\,against})\\
+0.79(\mathrm{Typical\,penalties\,against}).
\end{array}
$$

In the scatter plots in the following section, the points indicate the median values from the MCMC samples.

# Model Results

First off, here's a quick legend showing the team logos and names, in case you're not familiar with the CPL.

<img src="/assets/images/soccer/legend.png" width="200">

The data I am using was uploaded on September 5th. As of that date, the table was the following:

| Place |            Name | GP | Wins | Draws | Losses | GF | GA | GD | Points |
| - | ---------------- | -- | ---- | ----- | ------ | -- | -- | -- | ------ |
| 1 |  Atlético Ottawa | 23 |   11 |     7 |      5 | 27 | 23 |  4 |     40 |
| 2 |          Cavalry | 22 |   11 |     4 |      7 | 33 | 27 |  6 |     37 |
| 3 |            Forge | 22 |   11 |     3 |      8 | 40 | 21 | 19 |     36 |
| 4 |           Valour | 23 |   10 |     6 |      7 | 32 | 24 |  8 |     36 |
| 5 |          Pacific | 22 |   10 |     5 |      7 | 30 | 30 |  0 |     35 |
| 6 |      York United | 23 |    7 |     5 |     11 | 23 | 31 | -8 |     26 |
| 7 |    HFX Wanderers | 22 |    7 |     3 |     12 | 22 | 33 | -11 |      24 |
| 8 |         Edmonton | 23 |    3 |     7 |     13 | 26 | 44 | -18 |      16 |

First, let's look at the typical number of shots that our model predicts for each team:

![shots for versus shots against](/assets/images/soccer/shots.png)

The most clear outlier is Forge's shots against. They allow about 3 fewer shots per game than all other teams. All other teams are remarkably similar in terms of their shots against. Shots for displays much more spread between the teams with Valour, York United, and Cavalry taking the most shots and Pacific and Edmonton taking the fewest.

Next, let's look at shot quality:

![xg per shot for versus xg per shot against](/assets/images/soccer/xg.png)

Edmonton typically concedes the chances with the largest xG in the league, with an average xG per shot of about 0.12. Atlético Ottawa, who are on top of the table, concede chances with the lowest xG on average at about 0.085. In terms of attacking xG per shot, however, they are at the bottom of the league with just over 0.09 xG per shot. Forge and Pacific lead the league with about 0.12.

Finally, we can explore the typical xG for and against among the teams.

![xg for versus xg against](/assets/images/soccer/overall.png)

In this plot, overall team strength increases moving from the top left to the bottom right. The weakest team appears to be Edmonton, both in terms of xG against (about 1.5) and xG for (a bit below 1.0). Forge on the other hand, appears the strongest with about 1.4 xG for and 0.9 against. The current top team in the table, Atlético Ottawa, have the second smallest xG against, but also the second smallest xG for. Cavalry has a comparably strong attack as Forge FC with about 1.4 xG for, but has an average defence with abour 1.25 xG against.

It's interesting to note that this model knows nothing about the actual goals scored by any of the teams - only the numbers of shots and total xG for each game. Despite this, the plot of xG against versus xG for does a reasonable job of predicting the table. Three of the bottom four teams in the table (Edmonton, HFX Wanderers and Pacific), lie in the upper left of the plot. York United is performing worse in this table than this plot would suggest. The top four teams (Atlético Ottawa, Cavalry, Forge, Valour) are all among the bottom right 5 teams. This shows that xG is serving as a reasonable estimate of the actual scoring records of these teams.

One last thing that I'll explore in this post is the effect of home-field advantage. The effect of home-field advantage is very clear in the data. On average, a team will register about 20% more shots while playing at home than away. The quality of these shots, however, doesn't seem to be significantly different between home and away. This 20% boost means that Edmonton and Pacific at home would be expected to register a comparable number of shots to Atlético Ottawa or HFX Wanderers away. Even away from home, Forge would still be expected to allow fewer shots than any opponent.

Overall, this model suggests that York United is unfortunate to not be closer to the top 4 spots, which qualify for the playoffs. Atlético Ottawa is outperforming their xG to top the table, though several teams do have a game in hand. Forge has the clear strongest xG stats, and are likely favourites for this season despite their current table position. These results are interesting, but the real strength of this model should be in its ability to predict results! Come back soon where I'll use these results to try to predict the results of some upcoming matches.

The data in this post is courtesy of @CanPLdata.

#CCdata #CanPL

<img src="/assets/images/soccer/statsperform.png" width="200"><img src="/assets/images/soccer/centrecircle.png" width="200">

