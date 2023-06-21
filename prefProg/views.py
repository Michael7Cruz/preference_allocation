from django.shortcuts import render
from django.http import HttpResponse
from .forms import BudgetingForm
from .models import BudgetList
from .models import NativeItem
from .models import TypeDefinition
from .models import ComputationParam
from django.template import loader
from django.template.loader import render_to_string
from .computations import ComputeFinalWeights
import ast #converting from strings of suitable structure to multidimensional array

# Create your views here.
def home(request):
    if request.method == 'POST':
        typeFilter = request.POST.getlist('typecheck') #Get the list of activated radio buttons with typecheck as name.
        TypeFilteredList = TypeDefinition.objects.filter(id__in=typeFilter).values_list('name', flat=True)
        TypeList = TypeDefinition.objects.all
        NativeItemList = NativeItem.objects.all
        rendered = render_to_string('items_menu.html', {'menu':TypeFilteredList, 'types':TypeList, 'natives':NativeItemList}, request=request)
        return HttpResponse(rendered)
    else:
        TypeList = TypeDefinition.objects.all
        return render(request, 'home.html', {'types':TypeList})

def mainProg(request):
    allBudgetList = BudgetList.objects.all
    return render(request, 'mainProg.html', {'all':allBudgetList})

def budgeting(request):
    if request.method == 'POST':
        TypeList = TypeDefinition.objects.all()
        typeFilter = request.POST.getlist('typecheck') #Get the list of activated radio buttons with typecheck as name.
        TypeFilteredList = TypeDefinition.objects.filter(id__in=typeFilter).values_list('name', flat=True)
        itemFilter = request.POST.getlist('item_check') #Get the list of activated radio buttons with item_check as name. Value = ids
        NativeFilteredList = NativeItem.objects.filter(id__in=itemFilter).all() #Filter by id using itemFilter list
        hie = ComputationParam.objects.get(id=2)
        pric = ComputationParam.objects.get(id=3)
        
        weights = []    
        hierarchyStruc = []
        priceStruc = []
        if typeFilter != []: #Defines the hierarchy structure using a list
            
            counter1 = 0
            counter2 = 0
            for i in typeFilter:
                
                hierarchyStruc.append([TypeFilteredList[counter2],[]]) #id starts at 1 while list index starts at 0
                priceStruc.append([TypeFilteredList[counter2],[]])
                
                for j in NativeFilteredList.values():
                    if str(j["item_type_id"]) == str(i) and [j['item_subtype'],[]] not in hierarchyStruc[counter2][1]:   
                        hierarchyStruc[counter2][1].append([j['item_subtype'],[]])
                        priceStruc[counter2][1].append([j['item_subtype'],[]])
                    counter1 += 1  
                counter2 += 1
            
            
            counter2 = 0 
            for i in typeFilter:
                counter1 = 0
                temp = hierarchyStruc[counter2][1]
                for j in temp:
                    for k in NativeFilteredList.values():
                        if str(k["item_type_id"]) == str(i) and str(k['item_subtype']) == str(j[0]):
                            hierarchyStruc[counter2][1][counter1][1].append(k['name'])
                            priceStruc[counter2][1][counter1][1].append(k['price'])
                    counter1 += 1 
                counter2 += 1
                
            #Save it to a data field  
            hie.hierarchy_field = str(hierarchyStruc)
            hie.save()
            pric.hierarchy_field = str(priceStruc)
            pric.save()
                
        

        allocationForm = request.POST.getlist('allocation')
        if allocationForm != []: #If allocation form is not empty
            weights, immanence,_ = ComputeFinalWeights(ast.literal_eval(hie.hierarchy_field), ast.literal_eval(pric.hierarchy_field), allocationForm)
            
            rendered = render_to_string('computation.html', {'weights':weights, 'immanence':immanence,'hierac':hierarchyStruc,'menu':TypeFilteredList,'items':itemFilter, 'nativesFilter':NativeFilteredList, 'alloc':allocationForm}, request=request)
            return HttpResponse(rendered)
        else:
            return render(request, 'budgeting.html', {'hierac':hierarchyStruc,'menu':TypeFilteredList,'items':itemFilter, 'nativesFilter':NativeFilteredList})
    else:
        return render(request, 'budgeting.html', {})

