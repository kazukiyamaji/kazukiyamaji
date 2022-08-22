


console.log("hello  world????");
image_name = [];
noise_list = [];
search_result = [];
original = 0;
var load_image0=new Vue({
  el:"#load_image0",
  methods:{
    change_image:function(image){
      this.src=require(image)
      console.log(this.src)
    }
  }
})
var load_image1=new Vue({
  el:"#load_image1",
  methods:{
    change_image:function(image){
      console.log(image)
    }
  }
})
var load_image2=new Vue({
  el:"#load_image2",
  methods:{
    change_image:function(image){
      console.log(image)
    }
  }
})
var load_image3=new Vue({
  el:"#load_image3",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var load_image4=new Vue({
  el:"#load_image4",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var load_image5=new Vue({
  el:"#load_image5",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var load_image6=new Vue({
  el:"#load_image6",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var load_image7=new Vue({
  el:"#load_image7",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var load_image8=new Vue({
  el:"#load_image8",
  methods:{
    change_image:function(){
      console.log("change1")
    }
  }
})
var generator =new Vue({
  el:"#generator",
  data:{
    keyword:"hello"
  },
  methods:{
    post:function(){
      console.log("post!")
      config={
        headers:{
          'Content-Type':"application/json",
        }
      }
      console.log(this.keyword)
      data = { keyword: this.keyword }
      data= JSON.stringify(data);
      console.log(data)
      url="http://127.0.0.1:8001/display1/"
      axios.post(url,data,config)
      .then(function(response){
        noise_list = [];
        response=response.data
        for (i = 0; i <= 8; i++) {
          image_str = response.image["image" + String(i)].image;
          st = "#load_image" + String(i);
          
          eval("load_image"+i+".change_image(\"data:image/png;base64,\"+image_str)")
          $(st).off();
          $(st).on("click", function () {
            tmp = Number(this.id[10]);
            console.log(tmp);
            $("#number1").val(Number(this.id[10]));
          });
          document.getElementById("btn1").removeAttribute("disabled", true);
          document.getElementById("btn2").setAttribute("disabled", true);
          document.getElementById("btn3").setAttribute("disabled", true);
          image_name.push(response.image["image" + String(i)].filename);
        }
        search_result = response.result;
      })
    }
  }
})

$("#get_picture").on("submit", function (e) {
  page = Number($("#page").val());
  e.preventDefault();
  l = [];
  for (var i = 9 * page; i < 9 * (page + 1); i++) {
    l.push(search_result[i]);
  }
  data = { numbers: l };
  data = JSON.stringify(data);
  console.log(data)
  $.ajax({
    url: "http://127.0.0.1:8001/get_picture1/",
    type: "POST",
    dataType: "json",
    contentType: "application/json",
    data: data,
  }).done(function (response) {
    for (i = 0; i <= 8; i++) {
      image_str = response.image["image" + String(i)].image;
      st = "#load_image" + String(i);
      $(st).attr("src", "data:image/png;base64," + image_str);
      $(st).off();
      $(st).on("click", function () {
        tmp = Number(this.id[10]);
        console.log(tmp);
        $("#number1").val(Number(this.id[10]));
        document.getElementById("btn2").removeAttribute("disabled", true);
      });
      
      document.getElementById("btn2").setAttribute("disabled", true);
      document.getElementById("btn3").setAttribute("disabled", true);
      image_name.push(response.image["image" + String(i)].filename);
    }
  });
});
$("#invert").on("submit", function (e) {
  page = Number($("#page").val());
  num = Number($("#number1").val());
  e.preventDefault();
  pid = 9 * page + num;
  id = search_result[pid];
  console.log(id);
  if ($("#number1").val() != "") {
    data = { id: id };
    data = JSON.stringify(data);
    $.ajax({
      url: "http://127.0.0.1:8001/invert1/",
      type: "POST",
      dataType: "json",
      contentType: "application/json",
      data: data,
    }).done(function (response) {
      noise_list = [];
      for (i = 0; i <= 8; i++) {
        image_str = response.image["image" + String(i)]["image"];
        st = "#load_image" + String(i);
        original = response.image["image" + String(i)]["vec"];
        noise_list.push(response.image["image" + String(i)]["vec"]);
        $(st).attr("src", "data:image/png;base64," + image_str);
        $(st).off();
        $(st).on("click", function () {
          tmp = Number(this.id[10]);
          $("#number2").val(Number(this.id[10]));
          document.getElementById("btn3").removeAttribute("disabled", true);
        });
        document.getElementById("btn1").setAttribute("disabled", true);
        document.getElementById("btn2").setAttribute("disabled", true);
        document.getElementById("btn3").setAttribute("disabled", true);
      }
    });
  }
});
$("#select").on("submit", function (e) {
  num = $("#number2").val();
  e.preventDefault();
  if ($("#number2").val() != "") {
    console.log(noise_list);
    if ($("#same:checked").val() == "on") {
      vec = original;
    } else {
      vec = noise_list[num];
    }
    data = {
      vec: vec,
      label: $("#number2").val(),
      mode: $("#mode").val(),
      variation: $("#variation").val(),
      direction: $("#direction").val(),
    };
    data = JSON.stringify(data);
    $.ajax({
      url: "http://127.0.0.1:8001/cycle_generation1/",
      type: "POST",
      dataType: "json",
      contentType: "application/json",
      data: data,
    }).done(function (response) {
      noise_list = [];
      original = response.original;
      console.log(original);
      for (i = 0; i <= 8; i++) {
        image_str = response.image["image" + String(i)]["image"];
        st = "#load_image" + String(i);

        noise_list.push(response.image["image" + String(i)]["vec"]);
        $(st).attr("src", "data:image/png;base64," + image_str);
        $(st).off();
        $(st).on("click", function () {
          tmp = Number(this.id[10]);
          $("#number2").val(Number(this.id[10]));
        });
      }
    });
  }
});