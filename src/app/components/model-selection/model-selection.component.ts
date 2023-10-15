import { Component } from '@angular/core';

@Component({
  selector: 'app-model-selection',
  templateUrl: './model-selection.component.html',
  styleUrls: ['./model-selection.component.css']
})
export class ModelSelectionComponent {
 items: string [] = ['Linear Regression','CNN','ANN','Random Forest']
 selectedstring : string='';
}
