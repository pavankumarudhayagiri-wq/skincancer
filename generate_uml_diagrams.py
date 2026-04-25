from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle

OUT = Path(r"d:\skinlession cancer\Melanoma-Detection_App\report_assets")
OUT.mkdir(parents=True, exist_ok=True)

# 1) Use Case Diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')
ax.add_patch(Rectangle((2.2, 0.7), 7.6, 6.6, fill=False, linewidth=2, edgecolor='black'))
ax.text(6, 7.45, 'Melanoma Detection Web Application', ha='center', fontsize=12, fontweight='bold')

# Actors
ax.add_patch(Circle((0.9, 5.6), 0.18, fill=False, lw=1.8)); ax.plot([0.9,0.9],[5.42,4.9], color='black', lw=1.8)
ax.plot([0.62,1.18],[5.2,5.2], color='black', lw=1.8); ax.plot([0.9,0.62],[4.9,4.5], color='black', lw=1.8); ax.plot([0.9,1.18],[4.9,4.5], color='black', lw=1.8)
ax.text(0.9,4.2,'User',ha='center',fontsize=10)

ax.add_patch(Circle((11.1, 5.6), 0.18, fill=False, lw=1.8)); ax.plot([11.1,11.1],[5.42,4.9], color='black', lw=1.8)
ax.plot([10.82,11.38],[5.2,5.2], color='black', lw=1.8); ax.plot([11.1,10.82],[4.9,4.5], color='black', lw=1.8); ax.plot([11.1,11.38],[4.9,4.5], color='black', lw=1.8)
ax.text(11.1,4.2,'Admin',ha='center',fontsize=10)

use_cases = [
    (4.0,6.3,'Register / Login'),(6.1,6.3,'Upload Image'),(8.2,6.3,'Select Model'),
    (4.0,4.9,'View Prediction'),(6.1,4.9,'View Metrics'),(8.2,4.9,'View Visualizations'),
    (4.0,3.5,'Educational Resources'),(6.1,3.5,'Submit Feedback'),(8.2,3.5,'Logout')
]
for x,y,t in use_cases:
    ax.add_patch(FancyBboxPatch((x-0.95,y-0.28),1.9,0.56,boxstyle='round,pad=0.02',fill=False,lw=1.5))
    ax.text(x,y,t,ha='center',va='center',fontsize=9)

for x,y,_ in use_cases:
    if x<=6.1:
        ax.plot([1.2,x-0.95],[5.2,y], color='black', lw=1)
for x,y,_ in use_cases:
    if x>=6.1:
        ax.plot([10.8,x+0.95],[5.2,y], color='black', lw=1)

fig.savefig(OUT/'uml_use_case.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# 2) Class Diagram
fig, ax = plt.subplots(figsize=(12, 8))
ax.set_xlim(0, 12); ax.set_ylim(0, 8); ax.axis('off')

def cls(x,y,w,h,name,attrs,methods):
    ax.add_patch(Rectangle((x,y),w,h,fill=False,lw=1.8))
    ax.plot([x,x+w],[y+h-0.5,y+h-0.5],color='black',lw=1.2)
    ax.plot([x,x+w],[y+h-1.7,y+h-1.7],color='black',lw=1.2)
    ax.text(x+w/2,y+h-0.25,name,ha='center',va='center',fontsize=10,fontweight='bold')
    ax.text(x+0.08,y+h-0.75,'\n'.join(attrs),ha='left',va='top',fontsize=8)
    ax.text(x+0.08,y+h-1.95,'\n'.join(methods),ha='left',va='top',fontsize=8)

cls(0.7,4.6,2.4,2.8,'User',['+id','+username','+email'],['+register()','+login()'])
cls(3.7,4.6,2.8,2.8,'AuthService',['+users.db'],['+register_user()','+authenticate_user()'])
cls(7.2,4.6,3.6,2.8,'ModelManager',['+model_name','+image_type'],['+load_model()','+predict()'])
cls(1.8,0.9,3.4,2.8,'PredictionRequest',['+image','+selected_model'],['+validate_input()'])
cls(6.0,0.9,3.2,2.8,'PredictionResult',['+label','+confidence'],['+display_result()'])

ax.annotate('',xy=(3.7,5.9),xytext=(3.1,5.9),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.text(3.35,6.1,'uses',fontsize=8,ha='center')
ax.annotate('',xy=(7.2,5.9),xytext=(6.5,5.9),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.text(6.85,6.1,'calls',fontsize=8,ha='center')
ax.annotate('',xy=(7.8,4.6),xytext=(4.8,3.0),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.text(6.1,3.8,'processes',fontsize=8,ha='center')
ax.annotate('',xy=(7.6,2.3),xytext=(5.2,2.3),arrowprops=dict(arrowstyle='->',lw=1.4))
ax.text(6.4,2.5,'returns',fontsize=8,ha='center')

fig.savefig(OUT/'uml_class_diagram.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# 3) Sequence Diagram
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12); ax.set_ylim(0, 7); ax.axis('off')
parts = [('User',1.2),('UI (Streamlit)',3.8),('Auth (SQLite)',6.2),('Model (TF/Keras)',8.6),('Result View',10.8)]
for name,x in parts:
    ax.text(x,6.6,name,ha='center',fontsize=10,fontweight='bold')
    ax.plot([x,x],[0.8,6.2],linestyle='--',color='black',lw=1)

def msg(y, x1, x2, text):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops=dict(arrowstyle='->',lw=1.4))
    ax.text((x1+x2)/2,y+0.12,text,ha='center',fontsize=8)

msg(5.7,1.2,3.8,'Open app / Submit login')
msg(5.0,3.8,6.2,'authenticate_user()')
msg(4.4,6.2,3.8,'auth status')
msg(3.7,1.2,3.8,'Upload image + model')
msg(3.0,3.8,8.6,'preprocess + predict')
msg(2.3,8.6,3.8,'label + confidence')
msg(1.6,3.8,10.8,'Display result / charts')
msg(1.0,1.2,10.8,'Logout')

fig.savefig(OUT/'uml_sequence_diagram.png', dpi=180, bbox_inches='tight')
plt.close(fig)

# 4) Activity Diagram
fig, ax = plt.subplots(figsize=(11, 8))
ax.set_xlim(0, 11); ax.set_ylim(0, 9); ax.axis('off')

def rbox(x,y,w,h,t):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03',fill=False,lw=1.4))
    ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=9)

def decision(x,y,t):
    pts = [(x,y+0.5),(x+0.8,y),(x,y-0.5),(x-0.8,y)]
    ax.add_patch(plt.Polygon(pts,fill=False,lw=1.4))
    ax.text(x,y,t,ha='center',va='center',fontsize=8)

def arr(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',lw=1.2))

ax.add_patch(Circle((5.5,8.4),0.18,color='black'))
rbox(4.3,7.4,2.4,0.7,'Open Web Application')
decision(5.5,6.4,'Authenticated?')
rbox(1.2,5.4,2.8,0.7,'Register / Login')
rbox(4.3,5.0,2.4,0.7,'Open Dashboard')
rbox(4.1,4.0,2.8,0.7,'Upload Image')
decision(5.5,3.0,'Valid?')
rbox(7.3,2.6,2.8,0.7,'Select Model')
rbox(7.3,1.6,2.8,0.7,'Run Prediction')
rbox(3.8,1.2,3.4,0.7,'Show Result + Metrics')
rbox(3.9,0.3,3.2,0.7,'Feedback / Logout')
ax.add_patch(Circle((5.5,-0.2),0.18,color='black'))

arr(5.5,8.2,5.5,8.1); arr(5.5,7.4,5.5,6.9); arr(5.5,5.9,5.5,5.7)
arr(4.7,6.2,2.6,5.8); ax.text(3.4,6.1,'No',fontsize=8)
arr(2.6,5.4,4.3,5.35)
arr(5.5,5.0,5.5,4.7); arr(5.5,4.0,5.5,3.5); arr(6.3,3.0,7.3,2.95); ax.text(6.7,3.2,'Yes',fontsize=8)
arr(5.2,2.6,4.8,2.0); ax.text(4.4,2.7,'No',fontsize=8)
arr(8.7,2.6,8.7,2.3); arr(8.7,1.6,7.2,1.55); arr(5.5,1.2,5.5,1.0); arr(5.5,0.3,5.5,0.0); arr(5.5,0.0,5.5,-0.02)

fig.savefig(OUT/'uml_activity_diagram.png', dpi=180, bbox_inches='tight')
plt.close(fig)

print('Generated UML PNG files in', OUT)
