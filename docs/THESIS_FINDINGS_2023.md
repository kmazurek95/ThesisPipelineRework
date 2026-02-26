# Beyond Policy Influence: A Deeper Dive into the Factors Driving Advocacy Group Prominence

**Kaleb Mazurek** | MSc Social Science Research, University of Amsterdam | June 2023

---

## What This Study Is About

This thesis examines why certain advocacy organizations receive disproportionate attention from U.S. legislators during Congressional debates. Rather than trying to measure policy influence directly (a notoriously difficult thing to operationalize), I focus on **prominence**: the degree to which politicians invoke an organization as a credible voice or useful resource during floor speeches. Prominence is not the same as lobbying success or policy access. It reflects a softer form of power, one where an organization becomes the default reference point for a constituency or issue area in the eyes of policymakers.

The central questions are:
1. Why are some groups given prominence by some politicians and not others?
2. Why are some groups given more prominence on certain issues by some politicians than others?

## Data and Methods

The analysis pipeline collects nearly 78,000 Congressional Record documents from the 114th and 115th Congresses via the GovInfo API, then searches them for mentions of 5,447 national advocacy organizations drawn from the Washington Representative Study. This produced roughly 24,000 unique mention passages.

To distinguish prominent mentions from routine ones, I trained a supervised machine learning classifier (SVM with count vectorization) on 1,000 hand-coded passages. The coding scheme follows Fraussen (2018): a mention is prominent if a group's views are adopted by a policymaker, if the group is described as having a significant role, if it is used as a rhetorical resource, or if the speaker conveys the group's importance to the policy process. The classifier achieved an accuracy of ~81%, an F1-score of 0.79 for prominent mentions, and an ROC AUC of 0.72.

I then used generalized linear mixed-effects models (binomial family) with random intercepts for organization and issue area to test hypotheses about what drives prominence. The three-level structure (mention, interest group, issue area) accounts for the fact that a group's likelihood of being afforded prominence varies depending on both the organization itself and the policy context.

## Key Findings

### The landscape is heavily skewed

Of the 1,903 organizations that were mentioned at least once, only about 46% of all mentions were classified as prominent. The top 1% of groups accounted for roughly 31% of all prominent mentions, and the top 10% accounted for about 66%. Meanwhile, 749 organizations had zero prominent mentions and 3,421 organizations from the original list were never mentioned at all. Prominence, like other forms of political engagement, is concentrated among a small set of players.

### Issue salience works in unexpected directions

I expected that groups mentioned in high-salience policy areas would be more likely to receive prominent mentions. The results partly support this but with a twist. Groups in medium-salience areas were more likely to be afforded prominence (OR = 1.49), while groups in high-salience areas were actually less likely to be prominent (OR = 0.70). One possible explanation is that high-salience issues attract so many competing voices that the attention gets diluted, making it harder for any single organization to stand out. In medium-salience areas, the field may be less crowded and individual groups can more easily become the go-to reference.

### Organizational age does not predict prominence

Contrary to the hypothesis (and to what much of the institutionalization literature would suggest), older organizations were not significantly more likely to receive prominent mentions (OR = 0.998, p = 0.22). This challenges the assumption that longevity in the political landscape automatically translates to being treated as a preeminent voice. Whatever drives prominence, it is not simply a matter of having been around for a long time.

### Policy breadth shows a positive but insignificant effect

Organizations active across more policy areas were somewhat more likely to be prominently mentioned, consistent with the idea that a broader agenda creates more intersections with legislative debate. But the effect was not statistically significant (p = 0.14), so this remains suggestive rather than conclusive.

### External lobbyists matter, and not in the direction I expected

The most surprising finding concerned lobbying. I hypothesized that reliance on external lobbyists would not significantly increase prominence, following Grossman's (2012) argument that outsourcing advocacy signals weak internal leadership. The opposite turned out to be the case. Organizations that employed external lobbyists had significantly higher odds of being afforded prominence (p = 0.001). Professional intermediaries appear to play a meaningful role in positioning organizations within legislative discourse, a finding that warrants further investigation.

### Politician-level factors mostly did not hold up

The results for politician-specific variables were largely null or ran counter to expectations:

- **Seniority** had a small but significant *negative* effect on affording prominence (OR = 0.98, p < 0.001). More senior politicians were slightly less likely to prominently invoke interest groups, which contradicts the intuition that longer-serving members would have more established relationships with advocacy organizations.
- **Term status** (whether a politician was approaching re-election) showed a weakly positive effect that did not reach conventional significance (p = 0.07 for year before term end).
- **Bills sponsored** had essentially no effect (OR = 1.001, p = 0.47).
- **Issue overlap** between the politician's policy domain and the organization's domain was positive but not significant (p = 0.15).
- **Party affiliation**: Republicans were slightly less likely to afford prominence than Democrats (OR = 0.89, p = 0.047).

## What This Means

The overall picture is that prominence in legislative debate is driven more by organizational positioning and the structure of the policy environment than by the characteristics of individual politicians. The finding on external lobbyists is particularly noteworthy because it contradicts the theoretical expectation and suggests that professional advocacy infrastructure matters more for this form of success than internal organizational qualities like age or membership base.

The null findings are also informative. The fact that seniority, bill sponsorship, and issue overlap do not clearly predict prominence-affording behavior suggests that the politician side of the "audience dynamic" (Halpin & Fraussen, 2017) may be less systematic than theoretical models assume, or that the relevant politician-level factors have not yet been identified.

## Limitations

A few important caveats:

- The Washington Representative Study data is current only to 2011, meaning some organizational characteristics (especially lobbying expenditure) may not reflect the situation during the 114th-115th Congresses.
- The Google Trends-based salience measure is an indirect proxy for public attention, not a direct measure of public priorities. It is susceptible to short-term spikes from media events.
- About a third of observations could not be mapped to a policy area using the available committee and bill metadata, and were dropped from the analysis. While these appear to be randomly distributed, the data loss is not trivial.
- The classifier, while performing reasonably well, has an F1-score of 0.65 for non-prominent mentions, meaning it is better at identifying prominence than at ruling it out. Misclassification will introduce some noise into the dependent variable.

## References

The theoretical framework draws primarily on Halpin & Fraussen (2017) for the conceptualization of prominence and the "audience dynamic," Grossman (2012) for institutionalized pluralism and organizational-level predictors, and Ibenskas & Bunea (2021) for the politician-interest group linkage. The measurement approach follows Fraussen (2018), who was the first to use supervised machine learning to classify prominent mentions in legislative text.

Full bibliography available in the [thesis PDF](Thesis_UvA_Kaleb_Mazurek.pdf).

---

*For the pipeline rebuild results (expanded dataset, improved classifier), see [PIPELINE_FINDINGS.md](PIPELINE_FINDINGS.md). For a comparison of where the thesis and pipeline results align and diverge, see [THESIS_EXTENSION_NOTES.md](THESIS_EXTENSION_NOTES.md).*
