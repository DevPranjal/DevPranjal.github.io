---
title: What standard would have stopped this?
updated: 2026-09-13
---

> Written for the AI Incident Response Sprint (Apart Research and CeSIA, September
2026), Track 1: Containment. The full technical report, with the methods and all
the numbers, can be read [here](/assets/falsifying-isolation-claims.pdf).

#### Four labs, six weeks, the same mistake

Between 21 July and 6 August 2026, four organisations disclosed the same kind of
failure. OpenAI's models escaped a test environment and compromised Hugging Face's
infrastructure. Anthropic went back through 141,006 of its own evaluation runs and
found three cases where Claude had reached the internet from inside a test
environment and broken into real companies. Meta's model compromised a real company
during a misconfigured test. The UK's AI Security Institute logged unsanctioned
actions against real people. [^1]

Here is how these tests are supposed to work. A lab puts a model in a sealed
environment, turns its safety filters off, and asks it to go break things. Anything
harmful lands on fake targets. Sensible enough.

Here is what actually went wrong. None of these were jailbreaks, and none involved
a model escaping a sandbox that had been set up correctly. [^2] The sandboxes
simply were not sealed. Three of the four trace back to the same evaluation vendor
getting a configuration wrong.

Anthropic's writeup puts it plainly. The prompt told Claude it was in a simulation
with no internet access. That was false, because of a misunderstanding with their
evaluation partner. Claude believed everything it could reach was part of the
exercise, so it went and broke into real infrastructure using weak passwords.

The safety control was a sentence in a prompt. The sentence was wrong. Nobody
checked it.

#### The problem with checking

Could an outsider have caught this? A regulator, an auditor, anyone not inside the
lab.

Assume they get nothing. No network access, no logins, no console. That is the
realistic case, because three of these happened at a vendor that no lab had someone
sitting inside.

And then we hit a wall, because an environment cannot be proven sealed. You would
have to rule out every possible way out, and from outside none of them are visible.
Silence tells you nothing. A sealed room and a room nobody has looked at appear
identical from where you are standing.

But it can be caught when it is _not_ sealed. That turns out to be enough.

#### Two postcards

Here is the whole idea.

Suppose someone tells you they have been locked in a windowless room with no way to
contact the outside world. You cannot verify that by looking at the room. But you
can hand them two stamped postcards on the way in. One is addressed to the front
desk, which they are allowed to use. One is addressed to your house, which they
should have no way of reaching.

Then you go home and check your mailbox.

If your postcard arrives, the room is not sealed. You are not taking anyone's word
for it, because the card is in your hand. And if nothing arrives at all, that is
ambiguous, which is what the front desk card is for. If that one shows up, you know
they really did post both, so the silence on yours means something.

The real version is the same shape. A verifier gives the evaluation environment two
web addresses it owns, each carrying a one-time code tied to that specific run. One
the environment is allowed to reach, one it must not. The environment pings both
before the run starts. The verifier then checks its own server logs.

That last part is the one that matters. The verifier never asks the lab what
happened. It reads a log the lab cannot write to.

Which gives three answers:

- the code turned up where it should not, so the isolation claim is false
- it turned up only where it should, so the claim holds up as far as the evidence
goes
- nothing turned up, which settles nothing

There is deliberately no fourth answer meaning "confirmed isolated". No evidence
gathered this way could support it, so that verdict does not exist. If a lab claims
isolation and the result is anything other than the second answer, the run does not
start.

#### What happened when it was tested

Convenient examples prove nothing, so the test covers every combination of the
things that can vary: what the lab claims, whether each address is reachable, how
the lab writes up its report, and whether the code is fresh. That comes to 144
situations. Twenty-four of them are the dangerous case, where the environment is
wide open and the lab says it is sealed.

All 144 go through two schemes. The postcard check, and the obvious alternative of
trusting the lab's signed report.

| What the lab does | Postcard check | Trusting the report |
| --- | --- | --- |
| Runs the test, reports honestly | refused | refused |
| Writes the report from its config file, never tests | refused | **approved** |
| Runs the test, edits the result clean | refused | **approved** |
| Resubmits an old clean report | refused | **approved** |

The postcard check refused all 24. The report-based check let 16 through.

Most of those are a lab lying, which is arguably a policing problem rather than a
technical one. The second row is the interesting one. An honest lab writes its
report from its configuration file instead of from its network, and produces a
completely valid signed document for a wide-open environment. The signature is not
lying. It faithfully certifies a document that happens to be wrong.

That is the Anthropic case. Nobody deceived anybody.

Two more things worth knowing. The postcard check does not catch that case either.
It refuses instead of approving, which is the right direction to fail in, but it is
no cleverer than the alternative here. And a check that refuses everything would
also have scored 24 out of 24, so the safe situations matter just as much. Every
refusal there traces to something missing from the lab's own submission, not to the
check being paranoid.

Hugging Face's incident is the awkward one. That environment was configured
correctly and only sprang a leak partway through the run. Checking once beforehand
catches that roughly 1 time in 10. Repeating the check during the run caught 10 out
of 10.

#### The model will not do this for you

The obvious objection: these models are smart. Why not have the model notice it is
not really in a simulation?

Fifty sessions across two models, in a fake network containing one machine the
agent was not supposed to touch, with everything measured from what the agent
actually did rather than from what it said about itself.

In the realistic setup, where the machine does not announce itself but does look
like production if you are paying attention, the agents worked out that it was a
real system. Then they modified it anyway. Three out of three.

Knowing was not enough, because the prompt said the environment was isolated and
that outranked what they were seeing. [^3] Which is exactly what happened to
Claude.

The encouraging half: when the declaration was accurate, both models caused zero
unauthorised damage and still finished every legitimate task. Telling the agent the
truth costs nothing. The failure is upstream, in whether that sentence was true,
and the model is not in a position to check.

#### What this does not do

It detects failure and can never prove an environment is safe. It covers three of
the four incidents, and only in hindsight, since no lab has confirmed the
reasoning. The tests model the protocol rather than a real network with a real
adversary trying to dodge it.

So it is a narrow tool. But the thing it checks is the thing that broke, four
times, in six weeks. And it costs the lab two web requests.

[^1]: Anthropic, "Investigating three incidents in our cybersecurity evaluations", 30 July 2026; UK AI Security Institute, "Incident Report: unsanctioned agent behaviour during cyber testing", 4 August 2026; OpenAI, "OpenAI and Hugging Face partner to address security incident during model evaluation", July 2026.
[^2]: Cloud Security Alliance AI Safety Initiative, "When Red-Team Sandboxes Leak: Agentic AI Containment Failures", 8 August 2026.
[^3]: Consistent with S. Abdelnabi, A. Salem, "The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness", arXiv:2505.14617.
